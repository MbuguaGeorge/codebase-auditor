from pydantic import BaseModel
from typing import Literal


class FileToAudit(BaseModel):
    path: str
    priority: Literal["high", "medium", "low"]
    reason: str


class FileToSkip(BaseModel):
    path: str
    reason: str


class PlannerOutput(BaseModel):
    files_to_audit: list[FileToAudit]
    files_to_skip: list[FileToSkip]
    audit_scope: Literal["security", "architecture", "both"]
    scope_reasoning: str
    estimated_file_count: int
