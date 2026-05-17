import json
from typing import Optional

from agents.base import BaseAgent
from tracer.logger import TracerLogger
from schemas.planner_output import PlannerOutput
from config import settings


PLANNER_SYSTEM_PROMPT = """
You are a senior software security and architecture auditor.

Your job is to analyse a repository's file structure and produce a focused,
prioritised audit plan for a downstream code reviewer.

You will be given a list of files in a repository. Based on this structure you must:

1. Identify which files are highest risk and should be audited first
   (authentication, payment logic, API endpoints, config files, dependency files)

2. Identify which files can be skipped
   (test files, generated files, migrations, static assets, documentation)

3. Determine the audit scope based on what you see
   (security only, architecture only, or both)

4. Provide brief reasoning for your decisions

You must respond with valid JSON only. No explanation outside the JSON.
Do not wrap the JSON in markdown code blocks.
The JSON must match this exact structure:

{
  "files_to_audit": [
    {
      "path": "relative/path/to/file.py",
      "priority": "high" | "medium" | "low",
      "reason": "brief reason why this file is a priority"
    }
  ],
  "files_to_skip": [
    {
      "path": "relative/path/to/file.py",
      "reason": "why this file is being skipped"
    }
  ],
  "audit_scope": "security" | "architecture" | "both",
  "scope_reasoning": "brief explanation of why you chose this scope",
  "estimated_file_count": 0
}
"""


class PlannerAgent(BaseAgent):
    """
    Receives the full file map from directory_scanner.py and
    produces a structured audit plan consumed by the executor.

    The planner does not read file contents.
    It reasons purely from file names, paths, and extensions.
    """

    def __init__(self, logger: TracerLogger):
        super().__init__(logger)

    def run(self, file_map: list[dict]) -> PlannerOutput:
        """
        Main entry point called by the orchestrator.

        Parameters:
            file_map: list of file metadata dicts from directory_scanner.py
                      each dict contains: path, extension, size_bytes

        Returns:
            PlannerOutput: validated Pydantic model with the audit plan
        """
        user_message = self._build_user_message(file_map)

        # call the LLM via base class
        raw_response = self.call(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_message=user_message,
            tools=None,
            iteration=1,
        )

        # parse and validate the response
        return self._parse_response(raw_response)

    def _build_user_message(self, file_map: list[dict]) -> str:
        """
        Builds the user message sent to the model.
        """
        # count files by extension for context
        extension_counts: dict[str, int] = {}
        for file in file_map:
            ext = file.get("extension", "no extension")
            extension_counts[ext] = extension_counts.get(ext, 0) + 1

        # build the file list as a readable string
        file_list = "\n".join(
            f"{file['path']} ({file.get('size_bytes', 0)} bytes)" for file in file_map
        )

        # build extension summary
        ext_summary = "\n".join(
            f"{ext}: {count} files" for ext, count in sorted(extension_counts.items())
        )

        return f"""
                Repository file structure ({len(file_map)} total files):
                {file_list}

                File type breakdown:
                {ext_summary}

                Please produce an audit plan for this repository.
                """

    def _parse_response(self, raw_response: str) -> PlannerOutput:
        """
        Parses the model's JSON response into a validated PlannerOutput.

        Two things can go wrong here:
        1. The model returns invalid JSON
        2. The JSON does not match the PlannerOutput schema

        Both are caught and raise clear errors so you know exactly
        what went wrong and can fix the prompt if needed.
        """
        # clean the response in case the model wrapped it in markdown
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            # strip markdown code fences
            lines = cleaned.split("\n")
            # remove first line (```json or ```) and last line (```)
            cleaned = "\n".join(lines[1:-1])

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Planner returned invalid JSON.\n"
                f"Error: {e}\n"
                f"Raw response:\n{raw_response}"
            )

        try:
            return PlannerOutput(**data)
        except Exception as e:
            raise ValueError(
                f"Planner JSON does not match expected schema.\n"
                f"Error: {e}\n"
                f"Parsed data: {data}"
            )
