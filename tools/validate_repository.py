"""Fail CI if the public source repository contains private or generated files."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".csv",
    ".doc",
    ".docx",
    ".jpg",
    ".jpeg",
    ".npz",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".svg",
    ".xls",
    ".xlsx",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def main() -> None:
    violations: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden artifact: {relative}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"unexpected binary file: {relative}")
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                violations.append(f"{label} pattern: {relative}")
    if violations:
        raise SystemExit("Public-tree validation failed:\n" + "\n".join(violations))
    print(f"Validated {len(tracked_files())} tracked source files")


if __name__ == "__main__":
    main()
