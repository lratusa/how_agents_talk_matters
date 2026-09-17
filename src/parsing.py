"""Frozen deterministic parsers (prereg §10).

Rule (frozen): LAST regex occurrence wins; if no match -> None (scored
incorrect, logged as unparsed). Parsers never see gold labels.
"""
import re

from .datasets import normalize_value

RE_FINAL = re.compile(r"FINAL ANSWER:\s*(.+?)\s*$", re.MULTILINE)
RE_VERDICT = re.compile(r"REVIEW VERDICT:\s*(PEER CORRECT|PEER INCORRECT)", re.IGNORECASE)
RE_UPDATED = re.compile(r"YOUR UPDATED ANSWER:\s*(.+?)\s*$", re.MULTILINE)
RE_LETTER = re.compile(r"[A-J]")


def _last(regex, text):
    ms = regex.findall(text or "")
    return ms[-1] if ms else None


def parse_answer(text, mcq):
    """Parse the LAST 'FINAL ANSWER: <x>' line.

    MCQ: first standalone A-J letter in the captured tail (uppercased).
    Free-answer: normalized value string. No match -> None.
    """
    tail = _last(RE_FINAL, text)
    if tail is None:
        return None
    if mcq:
        m = RE_LETTER.search(tail.strip().upper())
        return m.group(0) if m else None
    return normalize_value(tail)


def parse_review_verdict(text):
    """LAST 'REVIEW VERDICT:' line -> 'PEER CORRECT' | 'PEER INCORRECT' | None."""
    v = _last(RE_VERDICT, text)
    return v.upper() if v else None


def parse_updated_answer(text, mcq):
    """LAST 'YOUR UPDATED ANSWER: <x>' line -> letter/value | None."""
    tail = _last(RE_UPDATED, text)
    if tail is None:
        return None
    if mcq:
        m = RE_LETTER.search(tail.strip().upper())
        return m.group(0) if m else None
    return normalize_value(tail)


def split_justification(text):
    """Justification = everything before the LAST FINAL ANSWER line (stripped)."""
    ms = list(RE_FINAL.finditer(text or ""))
    if not ms:
        return (text or "").strip()
    return (text or "")[:ms[-1].start()].strip()
