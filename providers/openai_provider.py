import json
from openai import OpenAI
from typing import Optional

from providers.base_provider import BaseProvider, ProviderResponse
from config import settings


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider.
    Wraps the OpenAI API and normalises responses
    into the standard ProviderResponse format.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def chat(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> ProviderResponse:

        # OpenAI takes system prompt as the first message
        full_messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        api_kwargs = {
            "model": settings.MODEL,
            "max_tokens": settings.MAX_TOKENS,
            "messages": full_messages,
        }
        if tools:
            # convert from Anthropic tool format to OpenAI format if needed
            api_kwargs["tools"] = self._normalise_tools(tools)
            api_kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**api_kwargs)

        choice = response.choices[0]
        finish = choice.finish_reason

        # normalise stop reason to standard set
        stop_reason_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }
        stop_reason = stop_reason_map.get(finish, "end_turn")

        text = choice.message.content

        return ProviderResponse(
            text=text,
            stop_reason=stop_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            tool_calls=self.extract_tool_calls_from_raw(response),
            raw_response=response,
        )

    def extract_tool_calls(self, response: ProviderResponse) -> list[dict]:
        return response.tool_calls

    def extract_tool_calls_from_raw(self, raw_response) -> list[dict]:
        """
        Extracts tool calls from the raw OpenAI response.
        Returns a normalised list of dicts with: id, name, input
        """
        tool_calls = []
        choice = raw_response.choices[0]

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments),
                    }
                )

        return tool_calls

    def build_tool_result_messages(
        self,
        response: ProviderResponse,
        tool_results: list[dict],
    ) -> list[dict]:
        """
        OpenAI format:
        - assistant message contains the tool_calls list
        - each tool result is its own separate message with role "tool"
        """
        messages = []

        raw_choice = response.raw_response.choices[0]

        # assistant message with tool calls
        messages.append(
            {
                "role": "assistant",
                "content": raw_choice.message.content,
                "tool_calls": raw_choice.message.tool_calls,
            }
        )

        # one message per tool result
        for result in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_use_id"],
                    "content": result["content"],
                }
            )

        return messages

    def _normalise_tools(self, tools: list[dict]) -> list[dict]:
        """
        Converts tools from Anthropic format to OpenAI format.

        Anthropic format:
        {
            "name": "read_file",
            "description": "...",
            "input_schema": { ... }
        }

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "...",
                "parameters": { ... }
            }
        }

        If the tool is already in OpenAI format (has "type": "function")
        it is returned unchanged.
        """
        normalised = []

        for tool in tools:
            # already in OpenAI format
            if tool.get("type") == "function":
                normalised.append(tool)
                continue

            # convert from Anthropic format
            normalised.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )

        return normalised
