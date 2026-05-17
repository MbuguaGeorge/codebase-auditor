from pydantic import BaseModel, field_validator
from typing import Literal, Optional
from schemas.executor_output import Finding


class ValidatedFinding(BaseModel):
    original_finding_title: str
    file_path:              str
    is_genuine:             bool
    false_positive_reason:  Optional[str] = None
    final_severity:         Literal["critical", "high", "medium", "low"]
    severity_reasoning:     str
    validated_description:  str
    suggested_fix:          str
    evidence:               Optional[str] = None

    @field_validator("final_severity", mode="before")
    @classmethod
    def normalise_severity(cls, v):
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


class InvestigationRequest(BaseModel):
    file_path: str
    finding_title: str
    question: str


class CriticRawResponse(BaseModel):
    """
    Intermediate model used to parse the raw model response.
    Converted into CriticOutput after the investigation loop completes.
    """

    validated_findings: list[ValidatedFinding]
    investigation_requests: list[InvestigationRequest] = []
    summary: str


class CriticOutput(BaseModel):
    """
    Final output of the critic agent.
    This is what the orchestrator receives and passes to report.py.
    """

    validated_findings: list[ValidatedFinding]
    investigation_rounds: int
    supplementary_findings: list[Finding] = []
    summary: str
    total_genuine: int
    total_false_positives: int

    def get_by_severity(
        self, severity: Literal["critical", "high", "medium", "low"]
    ) -> list[ValidatedFinding]:
        """
        Convenience method to filter genuine findings by severity.
        Used by report.py to group findings in the output report.
        """
        return [
            f
            for f in self.validated_findings
            if f.is_genuine and f.final_severity == severity
        ]
