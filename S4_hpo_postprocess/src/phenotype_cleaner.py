"""Lightweight text normalisation for phenotype phrases."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[\u2018\u2019\u201c\u201d]")

# Minimal abbreviation dictionary; can be extended via config override.
DEFAULT_ABBREVIATIONS: Dict[str, str] = {
    "c/o": "complains of",
    "abd": "abdominal",
    "cx": "cervix",
    "ptosis": "ptosis",  # placeholder for consistent case
    "sz": "seizure",
}


class PhenotypeCleaner:
    def __init__(self, extra_abbreviations: Dict[str, str] | None = None) -> None:
        self.abbreviations = {**DEFAULT_ABBREVIATIONS}
        if extra_abbreviations:
            self.abbreviations.update(extra_abbreviations)

    def clean_list(self, phrases: Iterable[str]) -> List[str]:
        return [self.clean_text(p) for p in phrases if isinstance(p, str)]

    def clean_text(self, phrase: str) -> str:
        text = phrase.strip()
        if not text:
            return text

        # normalise punctuation quotes
        text = PUNCT_RE.sub("'", text)
        text = WHITESPACE_RE.sub(" ", text)

        # expand simple abbreviations without altering negations or semantics
        tokens = text.split()
        expanded = [self.abbreviations.get(tok.lower(), tok) for tok in tokens]
        text = " ".join(expanded)

        return text
