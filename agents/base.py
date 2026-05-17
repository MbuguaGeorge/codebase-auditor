import time
from typing import Optional
from abc import ABC, abstractmethod

from providers.factory import get_provider
from providers.base_provider import BaseProvider
from tracer.logger import TracerLogger
from tracer.models import ToolCall


class BaseAgent(ABC):

    def __init__(self, logger: TracerLogger):
        self.logger = logger
        self.provider: BaseProvider = get_provider()

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def call(
        self,
        system_prompt: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        iteration: int = 1,
    ) -> str:

        messages = [{"role": "user", "content": user_message}]
        all_tool_calls: list[ToolCall] = []

        start_time = self.logger.before_call(
            agent_name=self.agent_name,
            system_prompt=system_prompt,
            user_message=user_message,
        )

        try:
            while True:
                response = self.provider.chat(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools,
                )

                if response.stop_reason == "end_turn":
                    self.logger.after_call(
                        start_time=start_time,
                        agent_name=self.agent_name,
                        system_prompt=system_prompt,
                        user_message=user_message,
                        raw_response=response.text,
                        stop_reason=response.stop_reason,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        tool_calls=all_tool_calls,
                        iteration=iteration,
                        success=True,
                    )
                    return response.text

                elif response.stop_reason == "tool_use":
                    tool_results = self._handle_tool_use(
                        response=response,
                        all_tool_calls=all_tool_calls,
                    )

                    # let the provider build the correctly formatted messages
                    new_messages = self.provider.build_tool_result_messages(
                        response=response,
                        tool_results=tool_results,
                    )
                    messages.extend(new_messages)

                elif response.stop_reason == "max_tokens":
                    self.logger.after_call(
                        start_time=start_time,
                        agent_name=self.agent_name,
                        system_prompt=system_prompt,
                        user_message=user_message,
                        raw_response=response.text,
                        stop_reason=response.stop_reason,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        tool_calls=all_tool_calls,
                        iteration=iteration,
                        success=True,
                        error_message="Response truncated: max_tokens reached",
                    )
                    return response.text

                else:
                    raise ValueError(f"Unexpected stop reason: {response.stop_reason}")

        except Exception as e:
            self.logger.log_error(
                agent_name=self.agent_name,
                system_prompt=system_prompt,
                user_message=user_message,
                start_time=start_time,
                error=e,
                iteration=iteration,
            )
            raise

    def _handle_tool_use(
        self,
        response,
        all_tool_calls: list[ToolCall],
    ) -> list[dict]:
        tool_results = []

        raw_tool_calls = self.provider.extract_tool_calls(response)

        for tc in raw_tool_calls:
            tool_name = tc["name"]
            tool_input = tc["input"]
            tool_use_id = tc["id"]

            tool_result, success, error_msg = self._execute_tool(
                tool_name=tool_name,
                tool_input=tool_input,
            )

            all_tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_result=tool_result,
                    success=success,
                    error_message=error_msg,
                )
            )

            tool_results.append(
                {
                    "tool_use_id": tool_use_id,
                    "content": tool_result if success else f"Error: {error_msg}",
                }
            )

        return tool_results

    def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
    ) -> tuple[str, bool, Optional[str]]:
        raise NotImplementedError(
            f"Agent '{self.agent_name}' received tool call '{tool_name}' "
            f"but does not implement _execute_tool()."
        )

    @property
    def agent_name(self) -> str:
        return self.__class__.__name__.replace("Agent", "").lower()
