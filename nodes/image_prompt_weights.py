"""Preserve user-authored ComfyUI numeric prompt weights."""

from __future__ import annotations

import re


WEIGHTED_TAG = re.compile(
    r"(?<!\\)\((?P<term>(?:\\.|[^()\\\r\n])+?):\s*"
    r"(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)"
)


def is_weighted_tag(value: str) -> bool:
    """A complete numeric weight is a tag even if its text is a sentence."""
    return bool(WEIGHTED_TAG.fullmatch(value.strip()))


def preserve_weighted_tags(source: str, output: str) -> str:
    """Restore exact authored weights without rewriting unrelated output text."""
    authored = list(WEIGHTED_TAG.finditer(source))
    if not authored:
        return output
    existing = list(WEIGHTED_TAG.finditer(output))
    replacements: list[tuple[int, int, str]] = []
    missing: list[str] = []
    used = set()
    for item in authored:
        exact = item.group()
        if exact in output or exact in missing:
            continue
        term = " ".join(item.group("term").split()).casefold()
        match = next((candidate for candidate in existing
                      if candidate.start() not in used
                      and " ".join(candidate.group("term").split()).casefold() == term), None)
        if match:
            used.add(match.start())
            replacements.append((match.start(), match.end(), exact))
        else:
            missing.append(exact)
    for start, end, exact in sorted(replacements, reverse=True):
        output = output[:start] + exact + output[end:]
    return ", ".join(missing) + (", " if missing and output else "") + output
