import json
from typing import Optional

from agents.base import BaseAgent
from tracer.logger import TracerLogger
from schemas.planner_output import PlannerOutput, FileToAudit
from schemas.executor_output import ExecutorOutput, Finding
from tools.file_reader import read_file
from config import settings


EXECUTOR_SYSTEM_PROMPT = """
You are a senior software security and architecture auditor performing a detailed code review.

Your job is to carefully read a source code file and identify real, specific issues.

For each file you will:
1. Read the file contents using the read_file tool
2. Analyse the code for security vulnerabilities and architectural problems
3. Report only genuine issues with specific evidence from the code

Categories to look for:

SECURITY:
- Hardcoded secrets, API keys, passwords, tokens
- SQL injection via string concatenation
- Missing input validation or sanitisation
- Insecure authentication or authorisation
- Sensitive data exposed in logs or error messages
- Outdated or vulnerable dependency usage
- Missing rate limiting on critical endpoints
- Insecure file handling

ARCHITECTURE:
- God classes or functions doing too many things
- Business logic mixed into route handlers
- Missing error handling or bare except clauses
- N+1 database query patterns
- Missing indexes on frequently queried fields
- No retry logic for external API calls
- Synchronous blocking calls in async contexts
- Hardcoded configuration that should be environment variables
- Missing or inadequate logging

IMPORTANT RULES:
- Only report issues you can see directly in the code
- Include the specific line or code snippet as evidence
- Do not guess or assume issues that are not visible
- Do not report style preferences as issues
- If a file has no issues, return an empty findings list

You must respond with valid JSON only. No explanation outside the JSON.
Do not wrap the JSON in markdown code blocks.

The JSON must match this exact structure:

{
  "file_path": "path/to/file.py",
  "findings": [
    {
      "issue_type": "security" | "architecture" | "performance",
      "title": "short descriptive title",
      "description": "detailed explanation of the issue",
      "evidence": "the specific code snippet or line that shows the issue",
      "line_hint": "approximate line number or range if identifiable",
      "raw_severity": "critical" | "high" | "medium" | "low",
      "suggested_fix": "specific actionable suggestion"
    }
  ],
  "files_checked": ["path/to/file.py"],
  "analysis_notes": "any overall notes about this file"
}
"""

# This is the tool definition in the format the LLM expects.
# When the model sees this it knows it can call read_file during its response.

FILE_READER_TOOL = {
    "name": "read_file",
    "description": (
        "Reads the contents of a source code file from the repository. "
        "Call this to read a file before analysing it. "
        "Always call this tool first before reporting any findings."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The relative path to the file to read",
            }
        },
        "required": ["file_path"],
    },
}


class ExecutorAgent(BaseAgent):
    """
    Iterates through the planner's file list, reads each file
    using the read_file tool, and identifies security and
    architectural issues.

    One LLM call per file. All findings are collected and
    returned as a flat list for the critic to review.
    """

    def __init__(self, logger: TracerLogger, repo_path: str):
        super().__init__(logger)
        # repo_path is needed by the file_reader tool to resolve
        # relative file paths to absolute paths on disk
        self.repo_path = repo_path

    def run(self, planner_output: PlannerOutput) -> ExecutorOutput:
        """
        Main entry point called by the orchestrator.

        Iterates through files_to_audit from the planner output.
        Processes high priority files first, then medium, then low.
        """
        all_findings: list[Finding] = []
        files_checked: list[str] = []
        files_with_errors: list[str] = []

        # sort files by priority: high first, then medium, then low
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_files = sorted(
            planner_output.files_to_audit,
            key=lambda f: priority_order.get(f.priority, 99),
        )

        for file_info in sorted_files:
            try:
                findings = self._audit_file(
                    file_info=file_info,
                    iteration=len(files_checked) + 1,
                )
                all_findings.extend(findings)
                files_checked.append(file_info.path)

            except Exception as e:
                # log the error but continue to the next file
                # one bad file should not stop the entire audit
                files_with_errors.append(file_info.path)
                print(f"[executor] error auditing {file_info.path}: {e}")
                continue

        return ExecutorOutput(
            findings=all_findings,
            files_checked=files_checked,
            files_with_errors=files_with_errors,
        )

    def reinvestigate(
        self,
        file_path: str,
        investigation_request: str,
    ) -> list[Finding]:
        """
        Called by the critic when it needs more information about a specific file.

        The critic provides an investigation_request describing what it
        wants the executor to look at more carefully. The executor re-reads
        the file with that focused context and returns additional findings.
        """
        focused_prompt = f"""
        Re-examine this file with focused attention on the following concern:

        {investigation_request}

        Use the read_file tool to read the file at: {file_path}

        Report only findings directly related to the concern above.
        Apply the same JSON format as your standard audit.
        """
        raw_response = self.call(
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
            user_message=focused_prompt,
            tools=[FILE_READER_TOOL],
            iteration=99,  # use 99 to mark reinvestigation calls in traces
        )

        result = self._parse_file_response(raw_response, file_path)

        for finding in result.findings:
            finding.file_path = file_path
            
        return result.findings

    def _audit_file(
        self,
        file_info: FileToAudit,
        iteration: int,
    ) -> list[Finding]:
        """
        Audits one file by calling the LLM with the file reader tool.

        The model will:
        1. Call read_file to get the file contents
        2. Analyse the contents
        3. Return findings as JSON

        The base class handles the tool use loop automatically.
        """
        user_message = f"""
        Audit the following file for security vulnerabilities and architectural issues.

        File to audit:
        Path:     {file_info.path}
        Priority: {file_info.priority}
        Reason:   {file_info.reason}

        Use the read_file tool to read the file contents first,
        then return your findings in the specified JSON format.
        """
        raw_response = self.call(
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
            user_message=user_message,
            tools=[FILE_READER_TOOL],
            iteration=iteration,
        )

        result = self._parse_file_response(raw_response, file_info.path)

        for finding in result.findings:
            finding.file_path = file_info.path

        return result.findings

    def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
    ) -> tuple[str, bool, Optional[str]]:
        """
        Routes tool calls from the model to the correct tool function.
        Overrides the base class method.
        """
        if tool_name == "read_file":
            file_path = tool_input.get("file_path", "")

            # resolve the relative path against the repo root
            import os

            absolute_path = os.path.join(self.repo_path, file_path)

            content, success, error = read_file(absolute_path)
            return content, success, error

        else:
            return (
                f"Unknown tool: {tool_name}",
                False,
                f"Tool '{tool_name}' is not registered in the executor",
            )

    def _parse_file_response(
        self,
        raw_response: str,
        file_path: str,
    ):
        """
        Parses the model's JSON response for one file audit.

        Handles markdown stripping and raises clear errors
        if the JSON is invalid or does not match the schema.
        """
        from schemas.executor_output import FileAuditResult

        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Executor returned invalid JSON for {file_path}.\n"
                f"Error: {e}\n"
                f"Raw response:\n{raw_response}"
            )

        try:
            return FileAuditResult(**data)
        except Exception as e:
            raise ValueError(
                f"Executor JSON does not match schema for {file_path}.\n"
                f"Error: {e}\n"
                f"Data: {data}"
            )
