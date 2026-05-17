import subprocess
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class VulnerablePackage:
    """
    Represents one vulnerable package found by the CVE checker.
    """

    package_name: str
    installed_version: str
    vulnerability_id: str  # CVE ID or GHSA ID
    severity: str  # CRITICAL, HIGH, MODERATE, LOW
    description: str
    fix_version: str | None  # version that fixes the issue, if known
    source_file: str  # which dependency file it came from


@dataclass
class CVECheckResult:
    """
    Result of running the CVE checker on a dependency file.
    """

    source_file: str
    vulnerable_packages: list[VulnerablePackage]
    total_vulnerabilities: int
    checked_packages: int
    success: bool
    error_message: str | None


def check_dependencies(repo_path: str) -> list[dict]:
    """
    Finds all dependency files in the repo and checks them for known CVEs.

    Looks for: requirements.txt, Pipfile, pyproject.toml, package.json
    Runs pip-audit on Python dependency files.
    Returns a flat list of all vulnerabilities found across all files.
    """
    repo_root = Path(repo_path).resolve()
    results: list[CVECheckResult] = []

    # find all dependency files
    dependency_files = _find_dependency_files(repo_root)

    if not dependency_files:
        return [
            {
                "source_file": "none found",
                "vulnerable_packages": [],
                "total_vulnerabilities": 0,
                "checked_packages": 0,
                "success": True,
                "error_message": "No dependency files found in repository",
            }
        ]

    for dep_file in dependency_files:
        result = _check_file(dep_file)
        results.append(result)

    return [asdict(r) for r in results]


def check_requirements_file(file_path: str) -> dict:
    """
    Checks a single requirements.txt file for CVEs.
    Called directly by the executor tool when auditing a specific file.
    """
    result = _check_file(Path(file_path))
    return asdict(result)


# ── Private helpers ───────────────────────────────────────────────────────────


def _find_dependency_files(repo_root: Path) -> list[Path]:
    """
    Walks the repo looking for dependency files.
    Returns a list of absolute paths to found files.
    """
    dependency_filenames = {
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "Pipfile",
        "pyproject.toml",
        "package.json",
    }

    found: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        # skip irrelevant directories
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {"node_modules", ".git", "venv", ".venv", "env"}
        ]

        for filename in filenames:
            if filename in dependency_filenames:
                found.append(Path(dirpath) / filename)

    return found


def _check_file(file_path: Path) -> CVECheckResult:
    """
    Runs pip-audit against one dependency file and parses the results.

    pip-audit is a tool from PyPA that checks Python packages against
    the Python Packaging Advisory Database (PyPA Advisory DB) and
    the Open Source Vulnerabilities database (OSV).

    Why subprocess:
        pip-audit is a command-line tool. We call it via subprocess
        and parse its JSON output. This is simpler than using its
        Python API directly and gives us clean structured output.
    """
    filename = file_path.name

    # only run pip-audit on Python dependency files
    # package.json requires npm audit which is a separate tool
    if filename == "package.json":
        return _check_npm_file(file_path)

    if filename not in {
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "Pipfile",
        "pyproject.toml",
    }:
        return CVECheckResult(
            source_file=str(file_path),
            vulnerable_packages=[],
            total_vulnerabilities=0,
            checked_packages=0,
            success=False,
            error_message=f"Unsupported dependency file type: {filename}",
        )

    # build pip-audit command
    # --format json gives structured output
    # --output - prints to stdout
    # -r specifies a requirements file
    command = [
        "pip-audit",
        "--format",
        "json",
        "--output",
        "-",
    ]

    # requirements files use -r flag
    # pyproject.toml and Pipfile are detected automatically
    if filename.startswith("requirements"):
        command.extend(["-r", str(file_path)])
    else:
        # run from the directory containing the file
        command.append(".")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=str(file_path.parent),
        )

        # pip-audit exits with code 1 if vulnerabilities are found
        # exits with code 0 if clean
        # any other exit code is an actual error
        if result.returncode not in (0, 1):
            return CVECheckResult(
                source_file=str(file_path),
                vulnerable_packages=[],
                total_vulnerabilities=0,
                checked_packages=0,
                success=False,
                error_message=(
                    f"pip-audit failed with exit code {result.returncode}.\n"
                    f"stderr: {result.stderr[:500]}"
                ),
            )

        # parse the JSON output
        return _parse_pip_audit_output(
            output=result.stdout,
            source_file=str(file_path),
        )

    except subprocess.TimeoutExpired:
        return CVECheckResult(
            source_file=str(file_path),
            vulnerable_packages=[],
            total_vulnerabilities=0,
            checked_packages=0,
            success=False,
            error_message="pip-audit timed out after 120 seconds",
        )

    except FileNotFoundError:
        return CVECheckResult(
            source_file=str(file_path),
            vulnerable_packages=[],
            total_vulnerabilities=0,
            checked_packages=0,
            success=False,
            error_message=(
                "pip-audit not found. " "Install it with: pip install pip-audit"
            ),
        )

    except Exception as e:
        return CVECheckResult(
            source_file=str(file_path),
            vulnerable_packages=[],
            total_vulnerabilities=0,
            checked_packages=0,
            success=False,
            error_message=f"Unexpected error: {e}",
        )


def _parse_pip_audit_output(output: str, source_file: str) -> CVECheckResult:
    """
    Parses the JSON output from pip-audit into a CVECheckResult.

    pip-audit JSON output format:
    {
      "dependencies": [
        {
          "name": "package-name",
          "version": "1.0.0",
          "vulns": [
            {
              "id": "GHSA-xxxx-xxxx-xxxx",
              "description": "...",
              "fix_versions": ["1.0.1"],
              "aliases": ["CVE-2023-xxxx"]
            }
          ]
        }
      ]
    }
    """
    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        return CVECheckResult(
            source_file=source_file,
            vulnerable_packages=[],
            total_vulnerabilities=0,
            checked_packages=0,
            success=False,
            error_message=f"Failed to parse pip-audit output: {e}",
        )

    dependencies = data.get("dependencies", [])
    checked_count = len(dependencies)
    vulnerable: list[VulnerablePackage] = []

    for dep in dependencies:
        package_name = dep.get("name", "unknown")
        version = dep.get("version", "unknown")
        vulns = dep.get("vulns", [])

        for vuln in vulns:
            vuln_id = vuln.get("id", "unknown")
            description = vuln.get("description", "No description available")
            fix_versions = vuln.get("fix_versions", [])
            aliases = vuln.get("aliases", [])

            # use the CVE alias if available, otherwise use the GHSA ID
            display_id = next((a for a in aliases if a.startswith("CVE-")), vuln_id)

            # pip-audit does not always provide severity
            # infer from the vulnerability ID prefix as a rough heuristic
            severity = _infer_severity(vuln_id, description)

            fix_version = fix_versions[0] if fix_versions else None

            vulnerable.append(
                VulnerablePackage(
                    package_name=package_name,
                    installed_version=version,
                    vulnerability_id=display_id,
                    severity=severity,
                    description=description[:500],  # truncate long descriptions
                    fix_version=fix_version,
                    source_file=source_file,
                )
            )

    return CVECheckResult(
        source_file=source_file,
        vulnerable_packages=vulnerable,
        total_vulnerabilities=len(vulnerable),
        checked_packages=checked_count,
        success=True,
        error_message=None,
    )


def _check_npm_file(file_path: Path) -> CVECheckResult:
    """
    Runs npm audit on a package.json file.
    Returns a CVECheckResult in the same format as pip-audit results.

    Requires npm to be installed. Silently returns an error result
    if npm is not available.
    """
    try:
        result = subprocess.run(
            ["npm", "audit", "--json"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(file_path.parent),
        )

        # npm audit exits with 1 if vulnerabilities found, 0 if clean
        if result.returncode not in (0, 1):
            return CVECheckResult(
                source_file=str(file_path),
                vulnerable_packages=[],
                total_vulnerabilities=0,
                checked_packages=0,
                success=False,
                error_message=f"npm audit failed: {result.stderr[:300]}",
            )

        return _parse_npm_audit_output(
            output=result.stdout,
            source_file=str(file_path),
        )

    except FileNotFoundError:
        return CVECheckResult(
            source_file=str(file_path),
            vulnerable_packages=[],
            total_vulnerabilities=0,
            checked_packages=0,
            success=False,
            error_message="npm not found. Install Node.js to check JS dependencies.",
        )

    except Exception as e:
        return CVECheckResult(
            source_file=str(file_path),
            vulnerable_packages=[],
            total_vulnerabilities=0,
            checked_packages=0,
            success=False,
            error_message=f"npm audit error: {e}",
        )


def _parse_npm_audit_output(output: str, source_file: str) -> CVECheckResult:
    """
    Parses npm audit JSON output into a CVECheckResult.
    """
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return CVECheckResult(
            source_file=source_file,
            vulnerable_packages=[],
            total_vulnerabilities=0,
            checked_packages=0,
            success=False,
            error_message="Failed to parse npm audit output",
        )

    vulnerabilities = data.get("vulnerabilities", {})
    metadata = data.get("metadata", {})
    checked_count = metadata.get("dependencies", {}).get("total", 0)
    vulnerable: list[VulnerablePackage] = []

    for pkg_name, vuln_data in vulnerabilities.items():
        severity = vuln_data.get("severity", "unknown").upper()
        via = vuln_data.get("via", [])

        # via can contain strings (direct) or dicts (transitive)
        for source in via:
            if isinstance(source, dict):
                vuln_id = source.get("url", "unknown").split("/")[-1]
                description = source.get("title", "No description")
                fix_version = None

                fix_available = vuln_data.get("fixAvailable")
                if isinstance(fix_available, dict):
                    fix_version = fix_available.get("version")

                vulnerable.append(
                    VulnerablePackage(
                        package_name=pkg_name,
                        installed_version=vuln_data.get("range", "unknown"),
                        vulnerability_id=vuln_id,
                        severity=severity,
                        description=description[:500],
                        fix_version=fix_version,
                        source_file=source_file,
                    )
                )

    return CVECheckResult(
        source_file=source_file,
        vulnerable_packages=vulnerable,
        total_vulnerabilities=len(vulnerable),
        checked_packages=checked_count,
        success=True,
        error_message=None,
    )


def _infer_severity(vuln_id: str, description: str) -> str:
    """
    Infers severity from the vulnerability ID and description.
    Used when pip-audit does not provide a severity rating directly.

    This is a rough heuristic, not a precise assessment.
    The critic agent will calibrate severity more carefully.
    """
    description_lower = description.lower()

    if any(
        word in description_lower
        for word in [
            "remote code execution",
            "rce",
            "arbitrary code",
            "authentication bypass",
            "sql injection",
            "command injection",
            "privilege escalation",
        ]
    ):
        return "CRITICAL"

    if any(
        word in description_lower
        for word in [
            "denial of service",
            "dos",
            "cross-site scripting",
            "xss",
            "path traversal",
            "directory traversal",
            "sensitive information",
            "credentials",
        ]
    ):
        return "HIGH"

    if any(
        word in description_lower
        for word in [
            "open redirect",
            "csrf",
            "information disclosure",
            "improper validation",
        ]
    ):
        return "MEDIUM"

    return "LOW"


def format_cve_results_for_agent(results: list[dict]) -> str:
    """
    Formats CVE check results as a readable string for the executor agent.

    The executor calls this to format CVE findings before including
    them in the user message to the model. This keeps the raw data
    structures out of the prompt and gives the model clean text to reason about.
    """
    if not results:
        return "No CVE check results available."

    lines = []

    for result in results:
        source = result.get("source_file", "unknown")
        success = result.get("success", False)
        vulns = result.get("vulnerable_packages", [])
        checked = result.get("checked_packages", 0)
        error = result.get("error_message")

        lines.append(f"\n## {source}")

        if not success:
            lines.append(f"  ⚠ Check failed: {error}")
            continue

        lines.append(f"  Checked {checked} packages")

        if not vulns:
            lines.append("  ✓ No known vulnerabilities found")
            continue

        lines.append(f"  ✗ Found {len(vulns)} vulnerabilities:\n")

        for vuln in vulns:
            fix = vuln.get("fix_version")
            fix_text = f"→ fix: upgrade to {fix}" if fix else "→ no fix available"
            lines.append(
                f"  [{vuln.get('severity', '?')}] "
                f"{vuln.get('package_name')} {vuln.get('installed_version')} "
                f"({vuln.get('vulnerability_id')})\n"
                f"  {vuln.get('description', '')[:120]}...\n"
                f"  {fix_text}\n"
            )

    return "\n".join(lines)
