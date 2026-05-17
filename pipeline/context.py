from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4

from schemas.planner_output import PlannerOutput
from schemas.executor_output import ExecutorOutput
from schemas.critic_output import CriticOutput


@dataclass
class PipelineContext:
    """
    Shared state object passed through the entire pipeline.

    Created at the start of orchestrator.run() and passed
    to each agent in sequence. Each agent reads what it needs
    and writes its output back into the context.

    By the time the pipeline finishes, the context contains
    a complete record of everything that happened in this audit run.

    Why a dataclass instead of a plain dict:
        Type safety. You get autocomplete and type checking.
        If you try to access a field that does not exist you get
        an AttributeError immediately instead of a silent None.
    """

    session_id: str = field(default_factory=lambda: str(uuid4()))
    # human readable name for this run, defaults to timestamp
    run_name: str = field(
        default_factory=lambda: f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )

    repo_path: str = ""
    # absolute path to the repo being audited

    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Each field is set by the corresponding stage in the pipeline.
    # Fields are None until that stage has run.

    # set by directory scanner before the planner runs
    file_map: Optional[list[dict]] = None

    # set by the planner agent
    planner_output: Optional[PlannerOutput] = None

    # set by the executor agent
    executor_output: Optional[ExecutorOutput] = None

    # set by the critic agent
    critic_output: Optional[CriticOutput] = None

    # path to the generated report file
    report_path: Optional[str] = None

    status: str = "pending"
    # pending → running → completed | failed

    error_message: Optional[str] = None
    # set if the pipeline fails at any stage

    failed_stage: Optional[str] = None
    # which stage failed: scanner | planner | executor | critic | report

    # Track how long each stage took independently.
    # Useful for understanding where time is spent.

    stage_durations_ms: dict = field(default_factory=dict)
    # example: {"scanner": 120, "planner": 4200, "executor": 38000, "critic": 12000}

    @property
    def is_complete(self) -> bool:
        """True if the pipeline finished successfully."""
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        """True if the pipeline failed at any stage."""
        return self.status == "failed"

    @property
    def duration_ms(self) -> Optional[int]:
        """Total pipeline duration in milliseconds."""
        if self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return int(delta.total_seconds() * 1000)

    @property
    def total_findings(self) -> int:
        """Total raw findings from the executor."""
        if self.executor_output is None:
            return 0
        return len(self.executor_output.findings)

    @property
    def genuine_findings(self) -> int:
        """Total genuine findings after critic validation."""
        if self.critic_output is None:
            return 0
        return self.critic_output.total_genuine

    @property
    def false_positives(self) -> int:
        """Total false positives removed by the critic."""
        if self.critic_output is None:
            return 0
        return self.critic_output.total_false_positives

    @property
    def files_audited(self) -> int:
        """Total files audited by the executor."""
        if self.executor_output is None:
            return 0
        return len(self.executor_output.files_checked)

    def mark_stage_start(self, stage: str) -> float:
        """
        Records the start time of a pipeline stage.
        Returns the start time as a float for duration calculation.
        """
        import time

        self.status = "in_progress"
        return time.time()

    def mark_stage_end(self, stage: str, start_time: float) -> None:
        """
        Records the duration of a completed pipeline stage.
        """
        import time

        duration_ms = int((time.time() - start_time) * 1000)
        self.stage_durations_ms[stage] = duration_ms

    def mark_failed(self, stage: str, error: Exception) -> None:
        """
        Marks the pipeline as failed at a specific stage.
        Records the stage name and error message.
        """
        self.status = "failed"
        self.failed_stage = stage
        self.error_message = str(error)
        self.completed_at = datetime.utcnow()

    def mark_completed(self) -> None:
        """Marks the pipeline as successfully completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()

    def summary(self) -> dict:
        """
        Returns a summary dict of the pipeline run.
        Used by main.py to print the final result
        and by the tracer to update the PipelineSession.
        """
        return {
            "session_id": self.session_id,
            "run_name": self.run_name,
            "repo_path": self.repo_path,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "files_scanned": len(self.file_map) if self.file_map else 0,
            "files_audited": self.files_audited,
            "total_findings": self.total_findings,
            "genuine_findings": self.genuine_findings,
            "false_positives": self.false_positives,
            "report_path": self.report_path,
            "failed_stage": self.failed_stage,
            "error_message": self.error_message,
            "stage_durations_ms": self.stage_durations_ms,
        }
