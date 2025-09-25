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

    return {
        "file": str(file_path),
        "created": created_match.group(1).strip() if created_match else None,
        "last_updated": updated_match.group(1).strip() if updated_match else None,
        "status": status_match.group(1).strip() if status_match else None,
    }


def main():
    markdown_files = list(Path(".").rglob("*.md"))

    headers = []
    for md_file in markdown_files:
        try:
            header = extract_header(md_file)
        except Exception as e:
            print(f"Error processing {md_file}: {e}")
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
