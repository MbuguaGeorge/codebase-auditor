import os
from pathlib import Path

# maximum file size the reader will return in full
# files larger than this are truncated with a note
MAX_FILE_SIZE_BYTES = 100_000  # 100kb

# maximum number of lines to return
# prevents very long files from filling the context window
MAX_LINES = 800

# encodings to try in order
# some codebases have files in non-UTF-8 encodings
ENCODINGS_TO_TRY = ["utf-8", "latin-1", "cp1252", "ascii"]


def read_file(file_path: str) -> tuple[str, bool, str | None]:
    """
    Reads a file from disk and returns its contents as a string.

    Called by the executor agent's _execute_tool() method when
    the model calls the read_file tool.
    """
    path = Path(file_path)

    if not path.exists():
        return ("", False, f"File not found: {file_path}")

    if not path.is_file():
        return ("", False, f"Path is not a file: {file_path}")

    # check file size before reading
    try:
        size_bytes = path.stat().st_size
    except OSError as e:
        return ("", False, f"Cannot stat file: {e}")

    content = None
    used_encoding = None

    # try each encoding until one works
    for encoding in ENCODINGS_TO_TRY:
        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
        except OSError as e:
            return ("", False, f"Cannot read file: {e}")

    if content is None:
        return (
            "",
            False,
            f"Cannot decode file with any supported encoding: {file_path}",
        )

    lines = content.splitlines()
    truncated = False
    truncation_note = ""

    # truncate by line count
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES]
        truncated = True
        truncation_note = (
            f"\n\n[TRUNCATED: file has more than {MAX_LINES} lines. "
            f"Showing first {MAX_LINES} lines only.]"
        )

    # truncate by size
    content = "\n".join(lines)
    if len(content.encode("utf-8")) > MAX_FILE_SIZE_BYTES:
        # truncate to MAX_FILE_SIZE_BYTES characters
        content = content[:MAX_FILE_SIZE_BYTES]
        truncated = True
        truncation_note = (
            f"\n\n[TRUNCATED: file exceeds {MAX_FILE_SIZE_BYTES} bytes. "
            f"Showing first {MAX_FILE_SIZE_BYTES} bytes only.]"
        )

    # add a header with file metadata
    # this gives the model useful context about the file
    header = (
        f"# File: {file_path}\n"
        f"# Size: {size_bytes} bytes | "
        f"Lines: {len(lines)} | "
        f"Encoding: {used_encoding}"
    )

    if truncated:
        header += f" | TRUNCATED"

    full_content = f"{header}\n{'─' * 60}\n\n{content}"

    if truncated:
        full_content += truncation_note

    return (full_content, True, None)


def read_file_lines(
    file_path: str,
    start_line: int,
    end_line: int,
) -> tuple[str, bool, str | None]:
    """
    Reads a specific range of lines from a file.

    Useful when the executor needs to re-examine a specific section
    of a large file without reading the whole thing again.
    """
    path = Path(file_path)

    if not path.exists():
        return ("", False, f"File not found: {file_path}")

    for encoding in ENCODINGS_TO_TRY:
        try:
            with open(path, "r", encoding=encoding) as f:
                all_lines = f.readlines()

            # convert to 0-indexed
            start = max(0, start_line - 1)
            end = min(len(all_lines), end_line)

            selected = all_lines[start:end]
            content = "".join(selected)

            result = (
                f"# File: {file_path} (lines {start_line}–{end_line})\n"
                f"{'─' * 60}\n\n{content}"
            )

            return (result, True, None)

        except UnicodeDecodeError:
            continue
        except OSError as e:
            return ("", False, f"Cannot read file: {e}")

    return ("", False, f"Cannot decode file: {file_path}")
