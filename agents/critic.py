import json
from typing import Optional

from agents.base import BaseAgent
from tracer.logger import TracerLogger
from schemas.executor_output import ExecutorOutput, Finding
from schemas.critic_output import (
    CriticOutput,
    ValidatedFinding,
    InvestigationRequest,
)
from config import settings


CRITIC_SYSTEM_PROMPT = """
You are a principal engineer performing a rigorous peer review of a junior auditor's findings.

Your job is to review each finding and make a final determination:
1. Is this a genuine issue or a false positive?
2. What is the correct final severity?
3. Does this need more investigation before you can decide?

For each finding apply this thinking:

FALSE POSITIVE indicators:
- The "secret" is actually a public constant or placeholder
- The "SQL injection" uses parameterised queries when looked at carefully
- The "missing auth" endpoint is actually a public endpoint by design
- The issue is a style preference not a real problem

GENUINE ISSUE indicators:
- Clear evidence in the code snippet provided
- The issue has a realistic exploitation path
- The fix would meaningfully improve security or reliability

SEVERITY CALIBRATION:
- Critical: exploitable in production, data breach risk, auth bypass
- High:     significant risk, likely to cause real problems under normal use
- Medium:   real issue but requires specific conditions or low impact
- Low:      worth fixing but low risk, mostly best practices

You MUST use ONLY these exact values for final_severity: critical, high, medium, low.
No other values are accepted. If unsure, default to low.

INVESTIGATION:
If you cannot make a confident determination from the evidence provided,
request more investigation. Be specific about what you want examined.
Do not request investigation for issues where the evidence is clear.

You must respond with valid JSON only. No explanation outside the JSON.
Do not wrap the JSON in markdown code blocks.

The JSON must match this exact structure:

{
  "validated_findings": [
    {
      "original_finding_title": "exact title from the executor finding",
      "file_path": "path/to/file.py",
      "is_genuine": true | false,
      "false_positive_reason": "explanation if is_genuine is false, else null",
      "final_severity": "critical" | "high" | "medium" | "low",
      "severity_reasoning": "why you assigned this severity",
      "validated_description": "your refined description of the issue",
      "suggested_fix": "the fix from the executor or your improved version",
      "evidence": "the specific code snippet that demonstrates the issue, or null if not applicable"
    }
  ],
  "investigation_requests": [
    {
      "file_path": "path/to/file.py",
      "finding_title": "title of the finding that needs more investigation",
      "question": "specific question or area to investigate further"
    }
  ],
  "summary": "overall assessment of the codebase quality"
}
"""


class CriticAgent(BaseAgent):
    """
    Reviews executor findings, removes false positives,
    calibrates severity, and requests further investigation
    when evidence is ambiguous.

    The investigation loop allows the critic to send specific
    questions back to the executor, receive additional findings,
    and incorporate them before producing the final report.

    Max investigation rounds is configurable to prevent infinite loops.
    """

    # maximum number of investigation rounds before forcing a final decision
    MAX_INVESTIGATION_ROUNDS = 2

    def __init__(self, logger: TracerLogger, executor):
        """
        executor: a reference to the ExecutorAgent instance.
                  the critic holds this so it can call
                  executor.reinvestigate() during the review loop.
        """
        super().__init__(logger)
        self.executor = executor

    def run(self, executor_output: ExecutorOutput) -> CriticOutput:
        """
        Main entry point called by the orchestrator.

        Runs the review loop:
        1. Send all findings to the critic model for initial review
        2. If the critic requests investigation, call executor.reinvestigate()
        3. Incorporate new findings and run the critic again
        4. Repeat up to MAX_INVESTIGATION_ROUNDS times
        5. Return the final validated findings
        """
        # start with the executor's findings
        current_findings = executor_output.findings

        # track all additional findings discovered during investigation
        supplementary_findings: list[Finding] = []

        investigation_round = 0

        while investigation_round <= self.MAX_INVESTIGATION_ROUNDS:
            user_message = self._build_user_message(
                findings=current_findings,
                supplementary=supplementary_findings,
                investigation_round=investigation_round,
            )

            raw_response = self.call(
                system_prompt=CRITIC_SYSTEM_PROMPT,
                user_message=user_message,
                tools=None,
                iteration=investigation_round + 1,
            )

            critic_result = self._parse_response(raw_response)

            # check if the critic wants more investigation
            if (
                critic_result.investigation_requests
                and investigation_round < self.MAX_INVESTIGATION_ROUNDS
            ):
                # send each investigation request to the executor
                new_findings = self._handle_investigation_requests(
                    requests=critic_result.investigation_requests,
                )

                # add new findings to the supplementary list
                supplementary_findings.extend(new_findings)

                # combine original and new findings for the next round
                current_findings = executor_output.findings + supplementary_findings

                investigation_round += 1

                # loop continues — critic will review again with new evidence

            else:
                # no more investigation needed or max rounds reached
                # build and return the final output

                if (
                    critic_result.investigation_requests
                    and investigation_round >= self.MAX_INVESTIGATION_ROUNDS
                ):
                    # max rounds reached — note this in the summary
                    critic_result.summary += (
                        f" [Note: {len(critic_result.investigation_requests)} "
                        f"investigation request(s) were not fulfilled "
                        f"due to reaching the maximum investigation limit.]"
                    )

                return CriticOutput(
                    validated_findings=critic_result.validated_findings,
                    investigation_rounds=investigation_round,
                    supplementary_findings=supplementary_findings,
                    summary=critic_result.summary,
                    total_genuine=sum(
                        1 for f in critic_result.validated_findings if f.is_genuine
                    ),
                    total_false_positives=sum(
                        1 for f in critic_result.validated_findings if not f.is_genuine
                    ),
                )

        # safety fallback — should never reach here
        raise RuntimeError("Critic investigation loop exited unexpectedly")

    def _build_user_message(
        self,
        findings: list[Finding],
        supplementary: list[Finding],
        investigation_round: int,
    ) -> str:
        """
        Builds the user message for the critic.

        On round 0 sends the original findings.
        On subsequent rounds includes both original and supplementary findings
        with a note explaining the round context.
        """
        # format findings as readable JSON
        findings_json = json.dumps([f.model_dump() for f in findings], indent=2)

        if investigation_round == 0:
            return f"""
            Please review the following findings from the code auditor.

            Total findings to review: {len(findings)}

            Findings:
            {findings_json}

            Review each finding carefully and produce your validated assessment.
            Request investigation for any findings where you need more evidence.
            """
        else:
            supplementary_json = json.dumps(
                [f.model_dump() for f in supplementary], indent=2
            )
            return f"""
            This is investigation round {investigation_round}.

            You previously requested additional investigation.
            Here are the supplementary findings from that investigation:

            {supplementary_json}

            Below are all findings including the originals and the new supplementary findings.
            Please produce your final validated assessment now.

            All findings:
            {findings_json}
            """

    def _handle_investigation_requests(
        self,
        requests: list[InvestigationRequest],
    ) -> list[Finding]:
        """
        Sends each investigation request to the executor and collects results.

        The executor's reinvestigate() method re-reads the specified file
        with focused attention on the critic's question.
        """
        all_new_findings: list[Finding] = []

        for request in requests:
            try:
                new_findings = self.executor.reinvestigate(
                    file_path=request.file_path,
                    investigation_request=request.question,
                )
                # tag each finding to indicate it came from reinvestigation
                for finding in new_findings:
                    finding.file_path = request.file_path

                all_new_findings.extend(new_findings)

            except Exception as e:
                print(f"[critic] investigation failed for {request.file_path}: {e}")
                continue

        return all_new_findings

    def _parse_response(self, raw_response: str):
        """
        Parses the critic's JSON response.
        Returns a CriticRawResponse (intermediate model before building CriticOutput).
        """
        from schemas.critic_output import CriticRawResponse

        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Critic returned invalid JSON.\n"
                f"Error: {e}\n"
                f"Raw response:\n{raw_response}"
            )

        # normalise severity values before Pydantic validation
        severity_map = {
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "moderate": "medium",
            "low": "low",
            "informational": "low",
            "info": "low",
            "none": "low",
            "minimal": "low",
        }

        for finding in data.get("validated_findings", []):
            raw_severity = finding.get("final_severity", "").lower().strip()
            finding["final_severity"] = severity_map.get(raw_severity, "low")

        try:
            return CriticRawResponse(**data)
        except Exception as e:
            raise ValueError(
                f"Critic JSON does not match expected schema.\n"
                f"Error: {e}\n"
                f"Data: {data}"
            )

    def _execute_tool(self, tool_name, tool_input):
        """
        The critic uses no tools.
        This override exists to give a clear error if a tool is accidentally called.
        """
        raise NotImplementedError(
            "The critic agent does not use any tools. "
            "If you see this error the system prompt may be misconfigured."
        )
