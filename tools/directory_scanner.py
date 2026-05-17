import os
from pathlib import Path
from dataclasses import dataclass, asdict


# These are never useful to audit and would waste tokens if included.

SKIP_DIRS = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "eggs",
    ".eggs",
    "htmlcov",
    ".tox",
    "migrations",  # Django migrations — generated code
    "alembic",  # Alembic migrations — generated code
    "static",  # static assets
    "media",  # uploaded media
    "coverage_html",
}

SKIP_EXTENSIONS = {
    # compiled / binary
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    # archives
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".rar",
    # images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".webp",
    # fonts
    ".ttf",
    ".woff",
    ".woff2",
    ".eot",
    # data files that are too large to be useful
    ".csv",
    ".parquet",
    ".pkl",
    ".h5",
    # lock files — not useful for auditing
    ".lock",
    # compiled frontend
    ".min.js",
    ".min.css",
}

SKIP_FILENAMES = {
    ".DS_Store",
    "Thumbs.db",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    "poetry.lock",
    "package-lock.json",
    "yarn.lock",
    "Pipfile.lock",
    ".env",
    ".env.local",
    ".env.production",
    ".env.staging",
    ".env.development",
}

# files that are always high priority regardless of location
HIGH_PRIORITY_FILENAMES = {
    "settings.py",
    "config.py",
    "auth.py",
    "authentication.py",
    "permissions.py",
    "security.py",
    "middleware.py",
    "models.py",
    "views.py",  # Django views — often contains business logic
    "urls.py",  # Django URLs — maps endpoints
    "serializers.py",  # DRF serializers — input validation
    "requirements.txt",
    "Pipfile",
    "pyproject.toml",
    "package.json",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".env.example",
}

# extensions considered high value for auditing
HIGH_VALUE_EXTENSIONS = {
    ".py",  # Python source
    ".js",  # JavaScript
    ".ts",  # TypeScript
    ".jsx",  # React
    ".tsx",  # React TypeScript
    ".go",  # Go
    ".rb",  # Ruby
    ".php",  # PHP
    ".java",  # Java
    ".rs",  # Rust
    ".env",  # environment files
    ".yaml",
    ".yml",  # config files
    ".toml",  # config files
    ".json",  # config and data
    ".sh",  # shell scripts
    ".bash",
}


@dataclass
class FileInfo:
    """
    Metadata about one file in the repository.
    This is what the directory scanner returns for each file.
    The planner receives a list of these.
    """

    path: str  # relative path from repo root
    extension: str  # file extension including the dot, e.g. ".py"
    size_bytes: int  # file size in bytes
    filename: str  # just the filename without directory
    is_high_priority: bool  # pre-flagged as high priority by the scanner


def scan_directory(repo_path: str) -> list[dict]:
    """
    Walks the repository directory and returns a list of file metadata dicts.

    Skips directories and files that are not useful for auditing.
    Pre-flags files that are known to be high priority.
    """
    repo_root = Path(repo_path).resolve()

    if not repo_root.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")

    if not repo_root.is_dir():
        raise ValueError(f"Repository path is not a directory: {repo_path}")

    files: list[FileInfo] = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        # modify dirnames in place to prevent os.walk from descending
        # into directories we want to skip
        # this is the correct way to skip directories with os.walk
        dirnames[:] = [
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            # skip specific filenames
            if filename in SKIP_FILENAMES:
                continue

            # skip hidden files
            if filename.startswith("."):
                continue

            full_path = Path(dirpath) / filename
            extension = full_path.suffix.lower()

            # skip unwanted extensions
            if extension in SKIP_EXTENSIONS:
                continue

            # skip files that end in .min.js or .min.css
            if filename.endswith(".min.js") or filename.endswith(".min.css"):
                continue

            # get file size — skip files that are too large to be useful
            # very large files (over 500kb) are usually generated or data files
            try:
                size_bytes = full_path.stat().st_size
            except OSError:
                continue

            if size_bytes > 500_000:
                continue

            # compute relative path from repo root
            relative_path = str(full_path.relative_to(repo_root))

            # determine if this file is high priority
            is_high_priority = (
                filename in HIGH_PRIORITY_FILENAMES
                or extension in HIGH_VALUE_EXTENSIONS
            )

            files.append(
                FileInfo(
                    path=relative_path,
                    extension=extension,
                    size_bytes=size_bytes,
                    filename=filename,
                    is_high_priority=is_high_priority,
                )
            )

    # sort: high priority first, then by extension, then alphabetically
    files.sort(
        key=lambda f: (
            0 if f.is_high_priority else 1,
            f.extension,
            f.path,
        )
    )

    # convert to list of dicts for easy serialisation
    return [asdict(f) for f in files]


def get_directory_summary(file_map: list[dict]) -> dict:
    """
    Returns a summary of the scanned directory.
    Useful for logging and debugging.
    """
    extension_counts: dict[str, int] = {}
    total_size = 0

    for file in file_map:
        ext = file.get("extension") or "no extension"
        extension_counts[ext] = extension_counts.get(ext, 0) + 1
        total_size += file.get("size_bytes", 0)

    return {
        "total_files": len(file_map),
        "total_size_bytes": total_size,
        "high_priority": sum(1 for f in file_map if f.get("is_high_priority")),
        "by_extension": dict(
            sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)
        ),
    }
