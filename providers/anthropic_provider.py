import anthropic
from typing import Optional

from providers.base_provider import BaseProvider, ProviderResponse
from config import settings


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude provider.
    Wraps the Anthropic API and normalises responses
    into the standard ProviderResponse format.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def chat(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> ProviderResponse:

        api_kwargs = {
            "model": settings.MODEL,
            "max_tokens": settings.MAX_TOKENS,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            api_kwargs["tools"] = tools

        response = self.client.messages.create(**api_kwargs)

        # normalise stop reason to a standard set
        stop_reason_map = {
            "end_turn": "end_turn",
            "tool_use": "tool_use",
            "max_tokens": "max_tokens",
        }
        stop_reason = stop_reason_map.get(response.stop_reason, "end_turn")

        # extract text from content blocks
        text = None
        for block in response.content:
            if hasattr(block, "text"):
                text = block.text
                break

        return ProviderResponse(
            text=text,
            stop_reason=stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            tool_calls=self.extract_tool_calls_from_raw(response),
            raw_response=response,
        )

    def extract_tool_calls(self, response: ProviderResponse) -> list[dict]:
        return response.tool_calls

    def extract_tool_calls_from_raw(self, raw_response) -> list[dict]:
        """
        Extracts tool calls from the raw Anthropic response.
        Returns a normalised list of dicts with: id, name, input
        """
        tool_calls = []
        for block in raw_response.content:
            if block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        return tool_calls

    def build_tool_result_messages(
        self,
        response: ProviderResponse,
        tool_results: list[dict],
    ) -> list[dict]:
        """
        Anthropic format:
        - assistant message contains the raw response content
        - user message contains all tool results as a list
        """
        messages = []

        # assistant message with the raw response content
        messages.append(
            {
                "role": "assistant",
                "content": response.raw_response.content,
            }
        )

        # single user message containing all tool results
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result["tool_use_id"],
                        "content": result["content"],
                    }
                    for result in tool_results
                ],
            }
        )

        return messages
