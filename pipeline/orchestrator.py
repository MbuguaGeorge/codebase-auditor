from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from pipeline.context import PipelineContext
from tracer.logger import TracerLogger
from tracer.models import PipelineSession
from tracer.storage import TracerStorage
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.critic import CriticAgent
from tools.directory_scanner import scan_directory, get_directory_summary
from output.report import generate_report
from config import settings


console = Console()


class Orchestrator:
    """
    Manages the full pipeline from repo path to final report.

    Creates one context and one logger per pipeline run.
    All agents share the same logger so all trace events
    link to the same session_id.
    """

    def __init__(self, verbose: bool = True):
        """
        verbose: if True, prints rich console output during the run.
                 set to False in tests to suppress output.
        """
        self.verbose = verbose
        self.storage = TracerStorage()

    def run(self, repo_path: str) -> PipelineContext:
        """
        Runs the full pipeline against a repository.

        Stages:
            1. Validate repo path
            2. Initialise context and tracer session
            3. Directory scan
            4. Planner agent
            5. Executor agent
            6. Critic agent
            7. Report generation
            8. Finalise tracer session
        """

        resolved_path = str(Path(repo_path).resolve())

        if not Path(resolved_path).exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

        if not Path(resolved_path).is_dir():
            raise ValueError(f"Repository path is not a directory: {repo_path}")

        context = PipelineContext(repo_path=resolved_path)

        logger = TracerLogger(
            session_id=context.session_id,
            verbose=self.verbose,
        )

        pipeline_session = PipelineSession(
            session_id=context.session_id,
            repo_path=resolved_path,
            started_at=context.started_at,
        )
        self.storage.save_session(pipeline_session)

        if self.verbose:
            self._print_header(context, resolved_path)

        try:

            # Stage 1: Directory Scanner
            context = self._run_scanner(context)

            # Stage 2: Planner Agent
            context = self._run_planner(context, logger)

            # Stage 3: Executor Agent
            context = self._run_executor(context, logger)

            # Stage 4: Critic Agent
            context = self._run_critic(context, logger)

            # Stage 5: Report Generation
            context = self._run_report(context)

            # mark pipeline as complete after every stage has finished
            context.mark_completed()

        except Exception as e:
            # any unhandled exception fails the pipeline
            # the stage-specific methods set context.failed_stage
            # before re-raising so we know where it broke
            if not context.is_failed:
                context.mark_failed("unknown", e)

        finally:
            # always update the session record regardless of outcome
            self._finalise_session(context, pipeline_session)

            # always print the session summary
            if self.verbose:
                logger.print_session_summary()
                self._print_footer(context)

        return context

    def _run_scanner(self, context: PipelineContext) -> PipelineContext:
        """
        Stage 1: Scans the repository directory tree.
        Populates context.file_map.
        """
        if self.verbose:
            self._print_stage_header("STAGE 1", "Directory Scanner")

        stage_start = context.mark_stage_start("scanner")

        try:
            file_map = scan_directory(context.repo_path)
            context.file_map = file_map

            context.mark_stage_end("scanner", stage_start)

            if self.verbose:
                summary = get_directory_summary(file_map)
                self._print_scanner_summary(summary)

            return context

        except Exception as e:
            context.mark_failed("scanner", e)
            if self.verbose:
                console.print(f"[red]✗ Scanner failed: {e}[/red]")
            raise

    def _run_planner(
        self,
        context: PipelineContext,
        logger: TracerLogger,
    ) -> PipelineContext:
        """
        Stage 2: Runs the planner agent.
        Populates context.planner_output.
        """
        if self.verbose:
            self._print_stage_header("STAGE 2", "Planner Agent")

        stage_start = context.mark_stage_start("planner")

        try:
            planner = PlannerAgent(logger=logger)
            planner_output = planner.run(file_map=context.file_map)

            context.planner_output = planner_output
            context.mark_stage_end("planner", stage_start)

            if self.verbose:
                self._print_planner_summary(planner_output)

            return context

        except Exception as e:
            context.mark_failed("planner", e)
            if self.verbose:
                console.print(f"[red]✗ Planner failed: {e}[/red]")
            raise

    def _run_executor(
        self,
        context: PipelineContext,
        logger: TracerLogger,
    ) -> PipelineContext:
        """
        Stage 3: Runs the executor agent.
        Populates context.executor_output.
        """
        if self.verbose:
            self._print_stage_header("STAGE 3", "Executor Agent")

        stage_start = context.mark_stage_start("executor")

        try:
            executor = ExecutorAgent(
                logger=logger,
                repo_path=context.repo_path,
            )
            executor_output = executor.run(planner_output=context.planner_output)

            context.executor_output = executor_output
            context.mark_stage_end("executor", stage_start)

            if self.verbose:
                self._print_executor_summary(executor_output)

            return context

        except Exception as e:
            context.mark_failed("executor", e)
            if self.verbose:
                console.print(f"[red]✗ Executor failed: {e}[/red]")
            raise

    def _run_critic(
        self,
        context: PipelineContext,
        logger: TracerLogger,
    ) -> PipelineContext:
        """
        Stage 4: Runs the critic agent.
        Populates context.critic_output.

        The critic receives a reference to the executor so it can
        call executor.reinvestigate() during the review loop.
        We create a new executor instance here (same logger, same repo)
        specifically for reinvestigation calls.
        """
        if self.verbose:
            self._print_stage_header("STAGE 4", "Critic Agent")

        stage_start = context.mark_stage_start("critic")

        try:
            # create a dedicated executor instance for reinvestigation
            # this keeps the critic's reinvestigation calls separate
            # from the original audit calls in the trace logs
            reinvestigation_executor = ExecutorAgent(
                logger=logger,
                repo_path=context.repo_path,
            )

            critic = CriticAgent(
                logger=logger,
                executor=reinvestigation_executor,
            )

            critic_output = critic.run(executor_output=context.executor_output)

            context.critic_output = critic_output
            context.mark_stage_end("critic", stage_start)

            if self.verbose:
                self._print_critic_summary(critic_output)

            return context

        except Exception as e:
            context.mark_failed("critic", e)
            if self.verbose:
                console.print(f"[red]✗ Critic failed: {e}[/red]")
            raise

    def _run_report(self, context: PipelineContext) -> PipelineContext:
        """
        Stage 5: Generates the final report.
        Populates context.report_path.
        """
        if self.verbose:
            self._print_stage_header("STAGE 5", "Report Generation")

        stage_start = context.mark_stage_start("report")

        try:
            report_path = generate_report(context)
            context.report_path = report_path
            context.mark_stage_end("report", stage_start)

            if self.verbose:
                console.print(f"[green]✓ Report saved to: {report_path}[/green]")

            return context

        except Exception as e:
            context.mark_failed("report", e)
            if self.verbose:
                console.print(f"[red]✗ Report generation failed: {e}[/red]")
            raise

    def _finalise_session(
        self,
        context: PipelineContext,
        pipeline_session: PipelineSession,
    ) -> None:
        """
        Updates the PipelineSession record in storage with final totals.
        Called in the finally block so it always runs.
        """
        token_summary = self.storage.get_token_summary(context.session_id)

        pipeline_session.completed_at = context.completed_at or datetime.utcnow()
        pipeline_session.status = context.status
        pipeline_session.error_message = context.error_message
        pipeline_session.total_input_tokens = token_summary.get("total_input_tokens", 0)
        pipeline_session.total_output_tokens = token_summary.get(
            "total_output_tokens", 0
        )
        pipeline_session.total_events = (
            token_summary.get("total_events", 0)
            if "total_events" in token_summary
            else len(self.storage.get_events_by_session(context.session_id))
        )
        pipeline_session.total_duration_ms = context.duration_ms or 0

        self.storage.update_session(pipeline_session)

    def _print_header(self, context: PipelineContext, repo_path: str) -> None:
        console.print(
            Panel(
                f"[bold white]Codebase Auditor[/bold white]\n"
                f"[dim]Session: {context.session_id}[/dim]\n"
                f"[dim]Repo:    {repo_path}[/dim]",
                style="blue",
                expand=False,
            )
        )

    def _print_stage_header(self, stage: str, name: str) -> None:
        console.print(
            f"\n[bold blue]{stage}[/bold blue] [bold white]{name}[/bold white]"
        )
        console.print("─" * 50)

    def _print_scanner_summary(self, summary: dict) -> None:
        console.print(
            f"[green]✓[/green] Found [cyan]{summary['total_files']}[/cyan] files "
            f"([cyan]{summary['high_priority']}[/cyan] high priority)"
        )

    def _print_planner_summary(self, planner_output) -> None:
        console.print(
            f"[green]✓[/green] Audit plan: "
            f"[cyan]{len(planner_output.files_to_audit)}[/cyan] files to audit, "
            f"[dim]{len(planner_output.files_to_skip)}[/dim] skipped"
        )
        console.print(f"  Scope: [yellow]{planner_output.audit_scope}[/yellow]")

    def _print_executor_summary(self, executor_output) -> None:
        console.print(
            f"[green]✓[/green] Audited "
            f"[cyan]{len(executor_output.files_checked)}[/cyan] files, "
            f"found [yellow]{len(executor_output.findings)}[/yellow] raw findings"
        )
        if executor_output.files_with_errors:
            console.print(
                f"  [yellow]⚠ {len(executor_output.files_with_errors)} "
                f"files could not be read[/yellow]"
            )

    def _print_critic_summary(self, critic_output) -> None:
        console.print(
            f"[green]✓[/green] Validated findings: "
            f"[red]{critic_output.total_genuine}[/red] genuine, "
            f"[dim]{critic_output.total_false_positives}[/dim] false positives removed"
        )
        if critic_output.investigation_rounds > 0:
            console.print(
                f"  Ran [cyan]{critic_output.investigation_rounds}[/cyan] "
                f"investigation round(s)"
            )

        # severity breakdown
        for severity in ["critical", "high", "medium", "low"]:
            count = len(critic_output.get_by_severity(severity))
            if count > 0:
                colour = {
                    "critical": "red",
                    "high": "yellow",
                    "medium": "cyan",
                    "low": "dim",
                }[severity]
                console.print(f"  [{colour}]{severity.upper()}: {count}[/{colour}]")

    def _print_footer(self, context: PipelineContext) -> None:
        if context.is_complete:
            console.print(
                Panel(
                    f"[bold green]✓ Audit Complete[/bold green]\n"
                    f"[dim]Duration:  {context.duration_ms}ms[/dim]\n"
                    f"[dim]Findings:  {context.genuine_findings} genuine "
                    f"({context.false_positives} false positives removed)[/dim]\n"
                    f"[dim]Report:    {context.report_path}[/dim]",
                    style="green",
                    expand=False,
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]✗ Audit Failed[/bold red]\n"
                    f"[dim]Stage:   {context.failed_stage}[/dim]\n"
                    f"[dim]Error:   {context.error_message}[/dim]",
                    style="red",
                    expand=False,
                )
            )
