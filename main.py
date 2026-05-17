import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from pipeline.orchestrator import Orchestrator


console = Console()


def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments.

    Usage:
        python main.py --repo /path/to/repo
        python main.py --repo /path/to/repo --quiet
        python main.py --repo /path/to/repo --output custom_report_name
    """
    parser = argparse.ArgumentParser(
        description="Codebase Auditor — AI-powered security and architecture review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python main.py --repo /path/to/my/project
        python main.py --repo ./my_project --quiet
        python main.py --repo ~/projects/myapp --output my_audit
        """,
    )

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path to the repository to audit (required)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress console output during the pipeline run",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom name for the output report (without extension)",
    )

    return parser.parse_args()


def validate_repo_path(repo_path: str) -> str:
    """
    Validates and resolves the repo path before passing to the orchestrator.
    Exits with a clear error message if the path is invalid.
    """
    path = Path(repo_path).resolve()

    if not path.exists():
        console.print(f"[red]Error: Repository path does not exist: {repo_path}[/red]")
        sys.exit(1)

    if not path.is_dir():
        console.print(
            f"[red]Error: Repository path is not a directory: {repo_path}[/red]"
        )
        sys.exit(1)

    return str(path)


def main() -> None:
    """
    Main entry point.

    Parses arguments, runs the pipeline, prints the result,
    and exits with the appropriate exit code.

    Exit codes:
        0 — audit completed successfully
        1 — audit failed or invalid arguments
    """
    args = parse_args()

    # validate the repo path early
    # before any expensive operations
    repo_path = validate_repo_path(args.repo)

    # create and run the orchestrator
    orchestrator = Orchestrator(verbose=not args.quiet)

    try:
        context = orchestrator.run(repo_path=repo_path)

    except KeyboardInterrupt:
        console.print("\n[yellow]Audit interrupted by user.[/yellow]")
        sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        sys.exit(1)

    # print final result and exit with appropriate code
    if context.is_complete:
        if not args.quiet:
            console.print(
                f"\n[bold green]Report saved to:[/bold green] {context.report_path}"
            )
        sys.exit(0)

    else:
        if not args.quiet:
            console.print(
                f"\n[bold red]Audit failed at stage:[/bold red] {context.failed_stage}"
            )
            console.print(f"[red]{context.error_message}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
