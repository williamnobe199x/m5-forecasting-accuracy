"""Remove internal claim labels from user-facing artifacts.

Internal confidence labels are useful inside Codex chat, but they should not
appear in webpages, markdown reports, documents, or other delivered artifacts.
This script is intentionally narrow and mechanical.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CLAIM_TAG_RE = re.compile(
    r"\[(?:KNOWN|COMPUTED|INFERRED|COMMON|FRAME|GUESS),\s*(?:HIGH|MED|LOW|VERY LOW|UNKNOWN)\]\s*"
)
AUDIT_LABEL = "RULES I " + "BROKE"
RULES_AUDIT_RE = re.compile(rf"^\s*\[{AUDIT_LABEL}\]:.*(?:\r?\n)?", re.MULTILINE)


DEFAULT_GLOBS = [
    "README.md",
    "DATA.md",
    "docs/**/*.md",
    "docs/**/*.html",
    "docs/**/*.js",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if claim tags remain.")
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--glob", action="append", dest="globs", help="Override artifact globs.")
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = CLAIM_TAG_RE.sub("", text)
    text = RULES_AUDIT_RE.sub("", text)
    return text


def artifact_paths(root: Path, globs: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in globs:
        for path in root.glob(pattern):
            if path.is_file():
                paths.add(path)
    return sorted(paths)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    globs = args.globs or DEFAULT_GLOBS
    changed: list[Path] = []
    remaining: list[Path] = []

    for path in artifact_paths(root, globs):
        text = path.read_text(encoding="utf-8")
        cleaned = clean_text(text)
        if cleaned != text:
            changed.append(path)
            if not args.check:
                path.write_text(cleaned, encoding="utf-8")
        if CLAIM_TAG_RE.search(cleaned) or RULES_AUDIT_RE.search(cleaned):
            remaining.append(path)

    if args.check and (changed or remaining):
        for path in changed or remaining:
            print(path.relative_to(root))
        raise SystemExit("claim tags remain in artifacts")

    for path in changed:
        print(f"cleaned {path.relative_to(root)}")
    if not changed:
        print("no claim tags found in artifacts")


if __name__ == "__main__":
    main()
