from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from uuid import uuid4


class ToolCall(BaseModel):
    """
    Represents a single tool call made by an agent during an LLM response.
    An agent can call multiple tools in one response, so this is a nested model
    used inside TraceEvent.
    """

    tool_name: str
    tool_input: dict
    tool_result: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class TokenUsage(BaseModel):
    """
    Tracks token consumption for one LLM API call.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TraceEvent(BaseModel):
    """
    A complete record of one LLM API call made by any agent in the pipeline.
    """

    # identity
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str

    # timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: Optional[int] = None

    # agent info
    agent_name: str
    iteration: int = 1

    # prompts sent to the model
    system_prompt: str
    user_message: str

    # response received from the model
    raw_response: Optional[str] = None
    stop_reason: Optional[str] = None

    # tools called during this event
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # token usage
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

    # status
    success: bool = True
    error_message: Optional[str] = None


class PipelineSession(BaseModel):
    """
    Represents one complete run of the full pipeline.
    Links all TraceEvents together under one session and tracks the
    overall pipeline outcome.
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    repo_path: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_events: int = 0
    total_duration_ms: int = 0

    status: str = "in_progress"  # in_progress, completed, failed
    error_message: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
