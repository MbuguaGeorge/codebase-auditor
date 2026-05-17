import os
from datetime import datetime
from pathlib import Path

from pipeline.context import PipelineContext
from schemas.critic_output import ValidatedFinding


REPORTS_DIR = Path(__file__).parent / "reports"


def generate_report(context: PipelineContext) -> str:
    """
    Generates a markdown audit report from the completed pipeline context.

    Called by the orchestrator after the critic agent finishes.
    Saves the report to output/reports/ and returns the file path.
    """
    # ensure reports directory exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # build report filename from run name and timestamp
    filename = f"{context.run_name}.md"
    report_path = REPORTS_DIR / filename

    # build the report content
    content = _build_report(context)

    # write to disk
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(report_path)


def _build_report(context: PipelineContext) -> str:
    """
    Builds the full markdown report as a string.
    Composed of multiple sections, each built by a helper function.
    """
    sections = [
        _build_header(context),
        _build_executive_summary(context),
        _build_severity_breakdown(context),
        _build_stage_timing(context),
        _build_findings_section(context),
        _build_false_positives_section(context),
        _build_files_section(context),
        _build_token_usage_section(context),
        _build_footer(context),
    ]

    return "\n\n".join(sections)


def _build_header(context: PipelineContext) -> str:
    repo_name = Path(context.repo_path).name
    timestamp = context.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    status = "❌ Failed" if context.is_failed else "✅ Completed"

    return f"""# Codebase Audit Report

    **Repository:** `{repo_name}`
    **Path:** `{context.repo_path}`
    **Session ID:** `{context.session_id}`
    **Generated:** {timestamp}
    **Status:** {status}"""


def _build_executive_summary(context: PipelineContext) -> str:
    critic = context.critic_output
    planner = context.planner_output

    if critic is None:
        return "## Executive Summary\n\n_Audit did not complete successfully._"

    scope = planner.audit_scope if planner else "unknown"

    # count by severity
    critical_count = len(critic.get_by_severity("critical"))
    high_count = len(critic.get_by_severity("high"))
    medium_count = len(critic.get_by_severity("medium"))
    low_count = len(critic.get_by_severity("low"))

    # determine overall risk level
    if critical_count > 0:
        risk_level = "🔴 **CRITICAL RISK**"
    elif high_count > 0:
        risk_level = "🟠 **HIGH RISK**"
    elif medium_count > 0:
        risk_level = "🟡 **MEDIUM RISK**"
    elif low_count > 0:
        risk_level = "🟢 **LOW RISK**"
    else:
        risk_level = "✅ **NO ISSUES FOUND**"

    return f"""## Executive Summary

    **Overall Risk:** {risk_level}

    This audit covered **{context.files_audited} files** with a scope of **{scope}**.
    The auditor identified **{context.genuine_findings} genuine issues**
    ({critical_count} critical, {high_count} high, {medium_count} medium, {low_count} low)
    after reviewing {context.total_findings} raw findings and removing
    {context.false_positives} false positives.

    {critic.summary}"""


def _build_severity_breakdown(context: PipelineContext) -> str:
    critic = context.critic_output
    if critic is None:
        return ""

    critical_count = len(critic.get_by_severity("critical"))
    high_count = len(critic.get_by_severity("high"))
    medium_count = len(critic.get_by_severity("medium"))
    low_count = len(critic.get_by_severity("low"))

    total = context.genuine_findings or 1  # avoid division by zero

    def bar(count: int, total: int, char: str = "█") -> str:
        """Builds a simple text progress bar."""
        filled = round((count / total) * 20)
        return char * filled + "░" * (20 - filled)

    return f"""## Severity Breakdown

    | Severity | Count | Distribution |
    |----------|-------|--------------|
    | 🔴 Critical | {critical_count} | `{bar(critical_count, total)}` |
    | 🟠 High     | {high_count}     | `{bar(high_count, total)}`     |
    | 🟡 Medium   | {medium_count}   | `{bar(medium_count, total)}`   |
    | 🟢 Low      | {low_count}      | `{bar(low_count, total)}`      |
    | Total   | {context.genuine_findings} | |

    **False positives removed:** {context.false_positives}
    **Investigation rounds:** {critic.investigation_rounds}"""


def _build_stage_timing(context: PipelineContext) -> str:
    durations = context.stage_durations_ms
    if not durations:
        return ""

    rows = []
    for stage, ms in durations.items():
        seconds = ms / 1000
        rows.append(f"| {stage.capitalize()} | {ms}ms | {seconds:.1f}s |")

    rows_text = "\n".join(rows)

    total_ms = context.duration_ms or sum(durations.values())

    return f"""## Pipeline Timing

    | Stage | Duration (ms) | Duration (s) |
    |-------|--------------|--------------|
    {rows_text}
    | **Total** | **{total_ms}ms** | **{total_ms/1000:.1f}s** |"""


def _build_findings_section(context: PipelineContext) -> str:
    critic = context.critic_output
    if critic is None or context.genuine_findings == 0:
        return "## Findings\n\n_No genuine issues found._"

    sections = ["## Findings\n"]

    # group by severity — critical first
    for severity in ["critical", "high", "medium", "low"]:
        findings = critic.get_by_severity(severity)
        if not findings:
            continue

        emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }[severity]

        sections.append(f"### {emoji} {severity.upper()} ({len(findings)} issues)\n")

        for i, finding in enumerate(findings, 1):
            sections.append(_format_finding(finding, i))

    return "\n".join(sections)


def _format_finding(finding: ValidatedFinding, index: int) -> str:
    """
    Formats one validated finding as a markdown block.
    """
    return f"""---

    #### {index}. {finding.original_finding_title}

    **File:** `{finding.file_path}`
    **Severity:** {finding.final_severity.upper()}

    **Description:**
    {finding.validated_description}

    **Evidence:**
    {finding.evidence if finding.evidence else 'See description above'}

    **Why this severity:** {finding.severity_reasoning}

    **Suggested Fix:**
    {finding.suggested_fix}

    """


def _build_false_positives_section(context: PipelineContext) -> str:
    critic = context.critic_output
    if critic is None or context.false_positives == 0:
        return ""

    false_positives = [f for f in critic.validated_findings if not f.is_genuine]

    if not false_positives:
        return ""

    rows = []
    for fp in false_positives:
        reason = fp.false_positive_reason or "No specific reason provided"
        rows.append(f"| `{fp.file_path}` | {fp.original_finding_title} | {reason} |")

    rows_text = "\n".join(rows)

    return f"""## False Positives Removed

    The critic agent identified and removed the following false positives:

    | File | Finding | Reason Dismissed |
    |------|---------|-----------------|
    {rows_text}"""


def _build_files_section(context: PipelineContext) -> str:
    executor = context.executor_output
    planner = context.planner_output

    if executor is None:
        return ""

    # files audited
    audited_rows = "\n".join(f"| `{f}` | ✅ Audited |" for f in executor.files_checked)

    # files skipped by planner
    skipped_rows = ""
    if planner and planner.files_to_skip:
        skipped_rows = "\n\n### Skipped Files\n\n"
        skipped_rows += "| File | Reason |\n|------|--------|\n"
        skipped_rows += "\n".join(
            f"| `{f.path}` | {f.reason} |" for f in planner.files_to_skip
        )

    # files with errors
    error_rows = ""
    if executor.files_with_errors:
        error_rows = "\n\n### Files With Errors\n\n"
        error_rows += "| File | Status |\n|------|--------|\n"
        error_rows += "\n".join(
            f"| `{f}` | ❌ Could not be read |" for f in executor.files_with_errors
        )

    return f"""## Files Audited

    | File | Status |
    |------|--------|
    {audited_rows}
    {skipped_rows}
    {error_rows}"""


def _build_token_usage_section(context: PipelineContext) -> str:
    """
    Pulls token usage from the tracer storage and includes it in the report.
    This is one of the most useful sections for understanding pipeline cost.
    """
    from tracer.storage import TracerStorage

    storage = TracerStorage()
    token_summary = storage.get_token_summary(context.session_id)

    if not token_summary:
        return ""

    by_agent = token_summary.get("by_agent", {})

    rows = []
    for agent_name, data in by_agent.items():
        rows.append(
            f"| {agent_name.capitalize()} | "
            f"{data['call_count']} | "
            f"{data['input_tokens']:,} | "
            f"{data['output_tokens']:,} | "
            f"{data['total_tokens']:,} |"
        )

    rows_text = "\n".join(rows)
    total_tokens = token_summary.get("total_tokens", 0)
    total_input = token_summary.get("total_input_tokens", 0)
    total_output = token_summary.get("total_output_tokens", 0)

    return f"""## Token Usage

    | Agent | API Calls | Input Tokens | Output Tokens | Total Tokens |
    |-------|-----------|-------------|---------------|-------------|
    {rows_text}
    | **Total** | | **{total_input:,}** | **{total_output:,}** | **{total_tokens:,}** |

    _Token usage determines API cost. The executor typically uses the most tokens
    since it makes one API call per file audited._"""


def _build_footer(context: PipelineContext) -> str:
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""---

    _Report generated by Codebase Auditor_
    _Session: `{context.session_id}`_
    _Generated at: {generated_at}_"""
