"""Company and fiscal year a question asks the search to stay inside."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Aliases are matched as whole words. Years are the filings in this corpus.
# The fiscal year is the calendar year in which that company's year ends.
_COMPANIES = (
    (("apple", "apples", "aapl"), "AAPL"),
    (("amazon", "amzn"), "AMZN"),
    (("microsoft", "msft"), "MSFT"),
    (("nvidia", "nvda"), "NVDA"),
    (("alphabet", "google", "googl", "goog"), "GOOGL"),
)
_YEAR_END_MONTH = {
    "AAPL": 9,  # last Saturday of September
    "MSFT": 6,  # June 30
    "AMZN": 12,  # December 31
    "GOOGL": 12,  # December 31
    "NVDA": 1,  # late January
}
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH = "|".join(_MONTHS)
_FISCAL = re.compile(r"\bfiscal(?:\s+year)?\s+(202[1-5])\b|\bfy\s*(202[1-5])\b", re.IGNORECASE)
_DATED = re.compile(rf"\b({_MONTH})\s+(?:\d{{1,2}},?\s+)?(202[1-5])\b", re.IGNORECASE)
_TRAILING_YEAR = re.compile(
    r"\s*(?:,(?:\s*and)?|and)\s*(202[1-5])\b", re.IGNORECASE
)
_HEADCOUNT = re.compile(
    r"\b(employee|employees|headcount|full-time|workforce|human capital)\b",
    re.IGNORECASE,
)
_SUFFIXES = (" Inc.", " Corporation", " Corp.", " Incorporated")


@dataclass(frozen=True, slots=True)
class FilingScope:
    ticker: str | None
    fiscal_year: int | None


def question_scopes(question: str) -> list[FilingScope]:
    """One scope per company and fiscal year a comparison asks about."""
    tickers = _tickers(question)
    points = _calendar_points(question)
    if not tickers and not points:
        return [FilingScope(None, None)]
    if not tickers:
        return [FilingScope(None, year) for year, _month in points]
    if not points:
        if len(tickers) == 1:
            return [FilingScope(tickers[0], None)]
        return [FilingScope(ticker, None) for ticker in tickers]
    scopes: list[FilingScope] = []
    for ticker in tickers:
        for year, month in points:
            scope = FilingScope(ticker, fiscal_year(ticker, year, month))
            if scope not in scopes:
                scopes.append(scope)
    return scopes


def fiscal_year(ticker: str, year: int, month: int | None) -> int:
    """Map a calendar date onto the fiscal year stored for this company.

    A month after the year-end belongs to the next fiscal year. An explicit
    fiscal year (month is None) is already the stored year.
    """
    if month is None:
        return year
    end = _YEAR_END_MONTH.get(ticker)
    if end is None or month <= end:
        return year
    return year + 1


def scope_query(question: str, scope: FilingScope) -> str:
    """Name the metric and that company's fiscal year-end for one search."""
    if scope.ticker is None or scope.fiscal_year is None:
        return question
    as_of = year_end_phrase(scope.ticker, scope.fiscal_year)
    if _HEADCOUNT.search(question):
        return f"full-time employees as of {as_of}. human capital. {question}"
    return f"{question} Fiscal year {scope.fiscal_year}, {as_of}."


def year_end_phrase(ticker: str, fiscal_year_number: int) -> str:
    """The year-end a 10-K uses for this fiscal year."""
    if ticker == "AAPL":
        return f"the last Saturday of September {fiscal_year_number}"
    if ticker == "MSFT":
        return f"June 30, {fiscal_year_number}"
    if ticker == "NVDA":
        return f"the end of January {fiscal_year_number}"
    return f"December 31, {fiscal_year_number}"


def filing_header(company_name: str, filing_type: str, fiscal_year: int) -> str:
    """Label a chunk with the filing it came from."""
    return f"{_short_company(company_name)} {filing_type} FY{fiscal_year},"


def _calendar_points(question: str) -> list[tuple[int, int | None]]:
    """Years in order. A month is set when the year came from a calendar date."""
    events: list[tuple[int, int, int | None, int]] = []
    for match in _FISCAL.finditer(question):
        year = int(match.group(1) or match.group(2))
        events.append((match.start(), year, None, match.end()))
    for match in _DATED.finditer(question):
        month = _MONTHS[match.group(1).lower()]
        year = int(match.group(2))
        events.append((match.start(), year, month, match.end()))
        rest = question[match.end() :]
        cursor = match.end()
        while (extra := _TRAILING_YEAR.match(rest)) is not None:
            events.append((cursor, int(extra.group(1)), month, cursor + extra.end()))
            rest = rest[extra.end() :]
            cursor += extra.end()
    events.sort()
    points: list[tuple[int, int | None]] = []
    seen: set[int] = set()
    for _start, year, month, _end in events:
        if year in seen:
            continue
        seen.add(year)
        points.append((year, month))
    return points


def _tickers(question: str) -> list[str]:
    lowered = question.lower()
    found: list[tuple[int, str]] = []
    for aliases, ticker in _COMPANIES:
        positions = [
            match.start()
            for alias in aliases
            if (match := re.search(rf"\b{re.escape(alias)}\b", lowered))
        ]
        if positions:
            found.append((min(positions), ticker))
    found.sort()
    tickers: list[str] = []
    for _, ticker in found:
        if ticker not in tickers:
            tickers.append(ticker)
    return tickers


def _short_company(name: str) -> str:
    cleaned = name.replace(",", "")
    for suffix in _SUFFIXES:
        cleaned = cleaned.removesuffix(suffix)
    return cleaned.removesuffix(".com").strip()
