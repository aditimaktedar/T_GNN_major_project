"""Inventory local files under data/raw, data/interim, and data/processed.

This utility does not download data, does not modify files, and does not treat
filenames as schemas.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from src.data.config import DATA_SEARCH_DIRS, PROJECT_ROOT
from src.data.inspect_data import _human_size, detect_file_type


def _safe_header_preview(path: Path, file_type: str, max_bytes: int = 4096) -> str | None:
    """Read a small header preview for delimited text files only."""
    if file_type not in {"csv", "tsv", "csv.gz", "tsv.gz"}:
        return None
    try:
        if file_type.endswith(".gz"):
            import gzip

            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                line = handle.readline(max_bytes)
        else:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                line = handle.readline(max_bytes)
    except OSError:
        return None
    return line.strip() or None


def inventory_file(path: Path, root: Path | None = None) -> dict:
    """Describe one file. Does not load the full contents."""
    stat = path.stat()
    file_type = detect_file_type(path)
    display_path = path
    if root is not None:
        try:
            display_path = path.relative_to(root)
        except ValueError:
            display_path = path

    record = {
        "path": str(display_path),
        "absolute_path": str(path.resolve()),
        "extension": path.suffix.lower() or "(none)",
        "file_type": file_type,
        "size_bytes": stat.st_size,
        "size_human": _human_size(stat.st_size),
        "empty": stat.st_size == 0,
        "header_preview": None,
    }
    if stat.st_size > 0:
        record["header_preview"] = _safe_header_preview(path, file_type)
    return record


def iter_data_files(directories: Sequence[Path]) -> Iterable[Path]:
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != ".gitkeep":
                yield path


def build_inventory(
    directories: Sequence[Path] | None = None,
    project_root: Path | None = None,
) -> list[dict]:
    """Return an inventory list for the given directories."""
    roots = tuple(directories) if directories is not None else DATA_SEARCH_DIRS
    root = project_root if project_root is not None else PROJECT_ROOT
    return [inventory_file(path, root=root) for path in iter_data_files(roots)]


def format_inventory(records: list[dict]) -> str:
    if not records:
        return (
            "Data inventory\n"
            "==============\n"
            "No data files found under data/raw, data/interim, or data/processed.\n"
            "Placeholder .gitkeep files are ignored.\n"
        )

    lines = [
        "Data inventory",
        "==============",
        f"Files found: {len(records)}",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"Path: {record['path']}",
                f"  Type: {record['file_type']}",
                f"  Extension: {record['extension']}",
                f"  Size: {record['size_human']} ({record['size_bytes']} bytes)",
                f"  Empty: {record['empty']}",
            ]
        )
        if record.get("header_preview"):
            lines.append(f"  Header preview: {record['header_preview']}")
        else:
            lines.append("  Header preview: (not available)")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List files under data/raw, data/interim, and data/processed."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the human-readable text report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = build_inventory()
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        print(format_inventory(records), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
