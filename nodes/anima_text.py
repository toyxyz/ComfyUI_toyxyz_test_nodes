"""Separate a user's Danbooru tag prefix from natural-language input."""

from __future__ import annotations

import re

from .image_prompt_weights import WEIGHTED_TAG, is_weighted_tag


_KOREAN = re.compile(r"[\uac00-\ud7a3]")
_SENTENCE_START = re.compile(
    r"^(?:a|an|the|she|he|they|we|i|this|that|there|her|his|their|"
    r"in|on|at|with|under|behind|before|after)\b", re.I)
_SENTENCE_VERB = re.compile(
    r"\b(?:is|are|was|were|stands?|sits?|walks?|wears?|holds?|looks?|"
    r"has|have|shows?|appears?|rests?|runs?|lies?|fills?|falls?|glows?)\b", re.I)
_TAG_THEN_SENTENCE = re.compile(r"\.\s+(?=(?:A|An|The|She|He|They|In|On|At|With)\b)")


def _prose_start(part: str) -> int | None:
    """Find a sentence start in one comma-delimited input component."""
    leading = len(part) - len(part.lstrip())
    content = part[leading:]
    if not content:
        return None
    if is_weighted_tag(content):
        return None
    line_break = re.search(
        r"\r?\n(?=\s*(?:(?:A|An|The|She|He|They|In|On|At|With)\b|[\uac00-\ud7a3]))",
        content)
    if line_break:
        return leading + line_break.end()
    if _KOREAN.search(content):
        return leading
    boundary = _TAG_THEN_SENTENCE.search(content)
    if boundary:
        return leading + boundary.end()
    if _SENTENCE_START.search(content) and _SENTENCE_VERB.search(content):
        return leading
    if len(content.split()) >= 6 and re.search(r"[.!?]", content):
        return leading
    return None


def _separate_weighted_tail(tags: str, prose: str) -> tuple[str, str]:
    """Keep standalone authored weights after prose in the source tag order.

    They are explicit tags, not generated camera guidance. Leaving them in
    prose lets the writer reorder them and the weight fallback prepend them.
    Do not extract a weight embedded in a sentence or quoted visible text.
    """
    def weights_only(remainder: str) -> bool:
        remainder = remainder.lstrip(" ,\t\r\n")
        while remainder:
            next_weight = WEIGHTED_TAG.match(remainder)
            if not next_weight:
                return False
            remainder = remainder[next_weight.end():].lstrip(" ,\t\r\n")
        return True

    matches = []
    for match in WEIGHTED_TAG.finditer(prose):
        before, after = prose[:match.start()], prose[match.end():]
        preceding_weight = bool(matches and
                                not prose[matches[-1].end():match.start()].strip(" ,\t\r\n"))
        standalone_start = (not before.strip() or
                            before.rstrip().endswith((",", ".", "!", "?")) or
                            before.endswith(("\n", "\r")) or preceding_weight)
        standalone_end = weights_only(after)
        if standalone_start and standalone_end:
            matches.append(match)
    if not matches:
        return tags, prose
    ordered = [tags.rstrip(" ,\r\n")] if tags.strip(" ,\r\n") else []
    ordered.extend(match.group() for match in matches)
    for match in reversed(matches):
        end = match.end()
        delimiter = re.match(r"[ \t]*,", prose[end:])
        if delimiter:
            end += delimiter.end()
        prose = prose[:match.start()] + prose[end:]
    return ", ".join(ordered), prose.strip(" ,\r\n")


def split_anima_input(source: str) -> tuple[str, str]:
    """Return authored tags in source order and the natural-language remainder."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Image prompter: enter an Anima prompt before running the queue.")
    segment_start = 0
    for comma in re.finditer(",", source):
        part = source[segment_start:comma.start()]
        offset = _prose_start(part)
        if offset is not None:
            return _separate_weighted_tail(
                source[:segment_start + offset], source[segment_start + offset:].strip())
        segment_start = comma.end()
    part = source[segment_start:]
    offset = _prose_start(part)
    if offset is not None:
        return _separate_weighted_tail(
            source[:segment_start + offset], source[segment_start + offset:].strip())
    return source, ""
