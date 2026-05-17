from pydantic import BaseModel, field_validator
from typing import Literal, Optional


class Finding(BaseModel):
    issue_type:    Literal["security", "architecture", "performance"]
    title:         str
    description:   str
    evidence:      Optional[str] = None
    line_hint:     Optional[str] = None
    raw_severity:  Literal["critical", "high", "medium", "low"]
    suggested_fix: str
    file_path:     Optional[str] = None

    @field_validator("line_hint", mode="before")
    @classmethod
    def coerce_line_hint_to_string(cls, v):
        """
        Coerces line_hint to a string regardless of what the model returns.
        The model sometimes returns an integer (e.g. 42) instead of a string.
        """
        if v is None:
            return None
        return str(v)

    @field_validator("raw_severity", mode="before")
    @classmethod
    def normalise_severity(cls, v):
        """
        Normalises severity values the model might return
        that fall outside the allowed literal set.
        """
        severity_map = {
            "critical":      "critical",
            "high":          "high",
            "medium":        "medium",
            "moderate":      "medium",
            "low":           "low",
            "informational": "low",
            "info":          "low",
            "none":          "low",
            "minimal":       "low",
        }
        if isinstance(v, str):
            return severity_map.get(v.lower().strip(), "low")
        return "low"


class FileAuditResult(BaseModel):
    """
    Result of auditing one file.
    Used internally by the executor to parse per-file responses.
    """

    file_path: str
    findings: list[Finding]
    files_checked: list[str]
    analysis_notes: Optional[str] = None


class ExecutorOutput(BaseModel):
    """
    Aggregated result of all files audited.
    This is what the executor returns to the orchestrator
    and what the critic receives as input.
    """

    findings: list[Finding]
    files_checked: list[str]
    files_with_errors: list[str] = []
