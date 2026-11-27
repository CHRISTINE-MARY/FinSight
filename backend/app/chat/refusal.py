"""Refuse before the model when the corpus or the retrieval cannot support the question.

The company check uses filing metadata. The evidence check uses retrieval
confidence and whether the retrieved filings cover the company that was asked.
Neither check looks for individual words in the passage text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.fusion import RRF_K
from app.retrieval.retriever import RetrievedPassage

_POSSESSIVE = re.compile(r"\b([A-Z][A-Za-z0-9]+)'s\b")
_INC = re.compile(r"\b([A-Z][A-Za-z0-9]+),?\s+Inc\.?")
_MULTI = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_NAME_STOP = {
    "com",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "ltd",
    "plc",
    "the",
}
# One list placing a chunk first. A lower score was never rank one anywhere.
_RANK_ONE = 1 / (RRF_K + 1)


@dataclass(frozen=True, slots=True)
class Filer:
    ticker: str
    company_name: str


def out_of_scope(question: str, filers: list[Filer]) -> bool:
    """The question names a company that is not a filer in the corpus."""
    if named_filers(question, filers):
        return False
    return bool(_organizations(question))


def evidence_sufficient(
    question: str, passages: list[RetrievedPassage], filers: list[Filer]
) -> bool:
    """Retrieved filings cover the asked company, or both searches agree."""
    if not passages:
        return False
    asked = named_filers(question, filers)
    if asked:
        tickers = {filer.ticker.casefold() for filer in asked}
        return any(
            passage.rrf_score > 0 and passage.ticker.casefold() in tickers
            for passage in passages
        )
    top = max(passages, key=lambda passage: passage.rrf_score)
    corroborated = top.semantic_rank is not None and top.lexical_rank is not None
    return corroborated and top.rrf_score + 1e-12 >= _RANK_ONE


def named_filers(question: str, filers: list[Filer]) -> list[Filer]:
    tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    return [filer for filer in filers if tokens & _filer_tokens(filer)]


def _filer_tokens(filer: Filer) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", filer.company_name.lower()))
    tokens -= _NAME_STOP
    tokens.add(filer.ticker.lower())
    return tokens


def _organizations(question: str) -> list[str]:
    found: list[str] = []
    for pattern in (_POSSESSIVE, _INC, _MULTI):
        for match in pattern.finditer(question):
            name = match.group(1)
            if name.casefold() in _NAME_STOP or name in found:
                continue
            found.append(name)
    return found
