import time
from datetime import datetime
from typing import Optional
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from tracer.models import TraceEvent, TokenUsage, ToolCall
from tracer.storage import TracerStorage


# one shared console instance used throughout the logger
console = Console()


class TracerLogger:
    """
    The core logger for the pipeline.

    Every agent holds a reference to this logger.
    Before an LLM call: call logger.before_call() to record the start time.
    After an LLM call:  call logger.after_call() to build and save the trace event.
    """

    def __init__(self, session_id: str, verbose: bool = True):
        self.session_id = session_id
        self.verbose = verbose
        self.storage = TracerStorage()

        # running totals updated after every call
        # useful for printing a live token count during development
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._event_count = 0

    def before_call(
        self,
        agent_name: str,
        system_prompt: str,
        user_message: str,
    ) -> float:
        """
        Called immediately before making an LLM API call.
        Records the start time so duration can be calculated after the call.

        Returns the start time as a float (unix timestamp).
        Pass this return value into after_call().
        """
        start_time = time.time()

        if self.verbose:
            console.print(
                f"\n[bold blue]→ [{agent_name.upper()}][/bold blue] "
                f"[dim]calling LLM...[/dim]"
            )

        return start_time

    def after_call(
        self,
        start_time: float,
        agent_name: str,
        system_prompt: str,
        user_message: str,
        raw_response: Optional[str],
        stop_reason: Optional[str],
        input_tokens: int,
        output_tokens: int,
        tool_calls: list[ToolCall],
        iteration: int = 1,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> TraceEvent:
        """
        Called immediately after an LLM API call completes.
        Builds a TraceEvent, saves it to storage, and prints a summary.

        Returns the TraceEvent so the agent can inspect it if needed.
        """
        # calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # build the token usage model
        token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # build the full trace event
        event = TraceEvent(
            session_id=self.session_id,
            agent_name=agent_name,
            iteration=iteration,
            system_prompt=system_prompt,
            user_message=user_message,
            raw_response=raw_response,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            token_usage=token_usage,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
        )

        # save to database
        self.storage.save_event(event)

        # update running totals
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._event_count += 1

        # print summary to console
        if self.verbose:
            self._print_event_summary(event)

        return event

    def log_error(
        self,
        agent_name: str,
        system_prompt: str,
        user_message: str,
        start_time: float,
        error: Exception,
        iteration: int = 1,
    ) -> TraceEvent:
        """
        Called when an LLM API call raises an exception.
        Records the failure as a TraceEvent so errors are visible in traces.
        """
        return self.after_call(
            start_time=start_time,
            agent_name=agent_name,
            system_prompt=system_prompt,
            user_message=user_message,
            raw_response=None,
            stop_reason=None,
            input_tokens=0,
            output_tokens=0,
            tool_calls=[],
            iteration=iteration,
            success=False,
            error_message=str(error),
        )

    def print_session_summary(self) -> None:
        """
        Prints a full summary of all events in this session.
        Call this at the end of the pipeline after all agents have finished.
        Shows total token usage, cost estimate, and per-agent breakdown.
        """
        token_summary = self.storage.get_token_summary(self.session_id)

        # main summary table
        table = Table(
            title=f"Pipeline Session Summary",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold blue",
        )

        table.add_column("Agent", style="bold")
        table.add_column("Calls", justify="right")
        table.add_column("Input Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("Total Tokens", justify="right")

        for agent_name, data in token_summary["by_agent"].items():
            table.add_row(
                agent_name,
                str(data["call_count"]),
                str(data["input_tokens"]),
                str(data["output_tokens"]),
                str(data["total_tokens"]),
            )

        # totals row
        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            str(self._event_count),
            str(token_summary["total_input_tokens"]),
            str(token_summary["total_output_tokens"]),
            f"[bold]{token_summary['total_tokens']}[/bold]",
        )

        console.print()
        console.print(table)
        console.print(f"\n[dim]Session ID: {self.session_id}[/dim]")

    def _print_event_summary(self, event: TraceEvent) -> None:
        """
        Prints a compact summary of one trace event to the console.
        Called after every LLM API call when verbose is True.
        """
        # choose colour based on agent
        agent_colours = {
            "planner": "green",
            "executor": "yellow",
            "critic": "red",
        }
        colour = agent_colours.get(event.agent_name, "white")

        # build the status line
        status = "[green]✓[/green]" if event.success else "[red]✗[/red]"

        lines = [
            f"{status} [bold {colour}]{event.agent_name.upper()}[/bold {colour}] "
            f"[dim]iteration {event.iteration}[/dim]",
            f"  tokens: [cyan]{event.token_usage.input_tokens}[/cyan] in / "
            f"[cyan]{event.token_usage.output_tokens}[/cyan] out / "
            f"[bold cyan]{event.token_usage.total_tokens}[/bold cyan] total",
            f"  duration: [magenta]{event.duration_ms}ms[/magenta]  "
            f"stop: [dim]{event.stop_reason or 'unknown'}[/dim]",
        ]

        # add tool calls if any were made
        if event.tool_calls:
            tool_names = [tc.tool_name for tc in event.tool_calls]
            lines.append(f"  tools called: [yellow]{', '.join(tool_names)}[/yellow]")

        # add error if failed
        if not event.success and event.error_message:
            lines.append(f"  [red]error: {event.error_message}[/red]")

        # add running total
        lines.append(
            f"  [dim]session total: "
            f"{self._total_input_tokens + self._total_output_tokens} tokens "
            f"across {self._event_count} calls[/dim]"
        )

        console.print("\n".join(lines))
