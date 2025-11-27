#!/usr/bin/env python3

import os
import re
from pathlib import Path

import pandas as pd
from tabulate import tabulate


def extract_header(file_path: Path) -> dict[str, str]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    created_match = re.search(r"-\s*\*\*Created\*\*:\s*(.+)", content)
    updated_match = re.search(r"-\s*\*\*Last Updated\*\*:\s*(.+)", content)
    status_match = re.search(r"-\s*\*\*Status\*\*:\s*`?([^`\n]+)`?", content)

    def _extract_match(match: re.Match[str] | None) -> str | None:
        return match.group(1).strip().strip('\\n",') if match else None

    return {
        "file": str(file_path),
        "created": _extract_match(created_match),
        "last_updated": _extract_match(updated_match),
        "status": _extract_match(status_match),
    }


def main():
    markdown_files = list(Path(".").rglob("*.md"))
    notebook_files = list(Path(".").rglob("*.ipynb"))
    files = markdown_files + notebook_files

    headers = []
    for f in files:
        try:
            header = extract_header(f)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue
        if header:
            headers.append(header)

    df = pd.DataFrame(headers).sort_values(by=["status", "last_updated", "created"])
    for status, group in df.groupby("status"):
        print(f"\n### Status: {status} ({len(group)}) ###")
        # print(group.drop(columns=["status"]).to_string(index=False))
        print(
            tabulate(
                group.drop(columns=["status"]),
                headers="keys",
                showindex=False,
                tablefmt="outline",
            )
        )


if __name__ == "__main__":
    main()
