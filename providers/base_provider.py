from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from tracer.models import ToolCall


@dataclass
class ProviderResponse:
    """
    Standardised response object returned by all providers.
    The base agent works with this regardless of which provider
    made the actual API call.
    """

    text: Optional[str]  # the model's text response
    stop_reason: str  # normalised: "end_turn" | "tool_use" | "max_tokens"
    input_tokens: int
    output_tokens: int
    tool_calls: list[dict]  # raw tool call data from the provider
    raw_response: object  # the original response object if needed


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Every provider must implement these three methods.
    """

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> ProviderResponse:
        """
        Makes one API call and returns a standardised ProviderResponse.
        """
        pass

    @abstractmethod
    def build_tool_result_messages(
        self,
        response: ProviderResponse,
        tool_results: list[dict],
    ) -> list[dict]:
        """
        Builds the messages to append after tool calls.
        Different providers format tool results differently.
        Returns a list of message dicts to extend the messages list with.
        """
        pass

    @abstractmethod
    def extract_tool_calls(
        self,
        response: ProviderResponse,
    ) -> list[dict]:
        """
        Extracts raw tool call data from the provider response.
        Returns a list of dicts with: name, input, id
        """
        pass
