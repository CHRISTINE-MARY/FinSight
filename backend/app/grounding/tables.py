"""Cut a stored table line and read the year column it belongs to.

The model names a row and a figure. The excerpt is the original line from the
chunk. A fiscal-year question then has to use the cell under that year, so a
2023 figure cannot pass as 2024 just because both sit on the same row.
"""

from __future__ import annotations

import re

_AMOUNT = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+")
_TOKEN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+\.\d+%?|\d+%|\d+")
_YEAR = re.compile(r"\b(?:fy\s*)?((?:19|20)\d{2})\b", re.IGNORECASE)
_YEAR_CELL = re.compile(r"(?:19|20)\d{2}")
_PERCENT = re.compile(r"\d+(?:\.\d+)?%")
_AMOUNT_THEN_YEAR = re.compile(
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+)\D{0,40}?((?:19|20)\d{2})"
)
_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
# A category table is often split so the year header sits two chunks above the row.
YEAR_HEADER_SPAN = 4


class GroundingError(Exception):
    """The answer cited something this request did not retrieve."""


def locate_line(content: str, label: str, figure: str, answer: str) -> str:
    """Return an exact slice of content. The full chunk is the fallback."""
    if not content:
        raise GroundingError("A citation is missing its excerpt")
    label = " ".join(label.split())
    figure = figure.strip()
    lines = [line for line in content.splitlines() if line]
    if not lines:
        raise GroundingError("A citation is missing its excerpt")
    number = _numeric_key(figure)
    if number and not any(_line_has_figure(line, figure) for line in lines):
        raise GroundingError("Locator is not in the retrieved passage")
    if label and not any(_line_has_label(line, label) for line in lines):
        raise GroundingError("Locator is not in the retrieved passage")

    matches = lines
    if label:
        matches = [line for line in matches if _line_has_label(line, label)]
    if number:
        matches = [line for line in matches if _line_has_figure(line, figure)]
    elif not label:
        amounts = amount_tokens(answer)
        narrowed = [
            line
            for line in matches
            if any(_line_has_figure(line, amount) for amount in amounts)
        ]
        if len(narrowed) == 1:
            matches = narrowed
        else:
            return content
    if not matches:
        raise GroundingError("Locator is not in the retrieved passage")
    if len(matches) > 1:
        if any(line.strip().startswith("|") for line in matches):
            raise GroundingError("Locator matches more than one row")
        return content
    return matches[0]


def locator_in_text(content: str, label: str, figure: str) -> bool:
    """True when this chunk contains the row name and the printed figure."""
    lines = [line for line in content.splitlines() if line]
    if not lines:
        return False
    number = _numeric_key(figure)
    if number and not any(_line_has_figure(line, figure) for line in lines):
        return False
    if label and not any(_line_has_label(line, label) for line in lines):
        return False
    return bool(number or label.strip())


def question_years(question: str) -> list[int]:
    years: list[int] = []
    for match in _YEAR.finditer(question):
        year = int(match.group(1))
        if year not in years:
            years.append(year)
    return years


def amount_tokens(text: str) -> list[str]:
    found: list[str] = []
    for match in _AMOUNT.finditer(text):
        token = match.group(0)
        if token not in found:
            found.append(token)
    return found


def reject_wrong_column(
    question: str,
    answer: str,
    cited: list[tuple[str, list[str], str]],
) -> None:
    """Check a table column, a percent, or a date against the stored text."""
    _reject_table_years(question, answer, cited)
    _reject_missing_amounts(answer, cited)
    _reject_percent_pairs(answer, cited)
    _reject_missing_date(question, answer, cited)


def _reject_table_years(
    question: str,
    answer: str,
    cited: list[tuple[str, list[str], str]],
) -> None:
    """A table figure has to sit in the year column the answer assigns it."""
    requested = question_years(question)
    pairs = _amount_year_pairs(answer)
    if len(requested) == 1 and not pairs:
        _reject_first_amount(requested[0], answer, cited)
        return
    for amount, year in pairs:
        columns = _columns_for(amount, cited)
        if columns and year not in columns:
            other = min(columns)
            raise GroundingError(f"{amount} is in the {other} column, not {year}")
    if len(requested) < 2 or pairs:
        return
    allowed = set(requested)
    for amount in amount_tokens(answer):
        columns = _columns_for(amount, cited)
        if columns and not columns & allowed:
            other = min(columns)
            raise GroundingError(
                f"{amount} is in the {other} column, not a requested year"
            )


def _reject_first_amount(
    year: int, answer: str, cited: list[tuple[str, list[str], str]]
) -> None:
    amounts = amount_tokens(answer)
    if not amounts:
        return
    first = amounts[0]
    found = _columns_for(first, cited)
    saw_header = _saw_year_header(cited)
    if not saw_header:
        return
    if year in found:
        return
    if found:
        other = min(found)
        raise GroundingError(f"{first} is in the {other} column, not {year}")
    raise GroundingError(f"{first} is not in the {year} column")


def _reject_percent_pairs(
    answer: str, cited: list[tuple[str, list[str], str]]
) -> None:
    """A percent has to appear in the chunk, with the year it is tied to."""
    haystack = _chunk_text(cited)
    percents = _PERCENT.findall(answer)
    if not haystack or not percents:
        return
    for percent in percents:
        if not _text_has_percent(haystack, percent):
            raise GroundingError(f"{percent} is not in the cited passage")
    answer_pairs = _percent_year_pairs(answer)
    source_pairs = _percent_year_pairs(haystack)
    if not answer_pairs or not source_pairs:
        return
    for percent, year in answer_pairs:
        if (percent, year) not in source_pairs:
            raise GroundingError(
                f"{percent}% is not the fiscal {year} figure in the cited passage"
            )


def _reject_missing_amounts(
    answer: str, cited: list[tuple[str, list[str], str]]
) -> None:
    """A printed figure in the answer has to occur in the cited chunk.

    Rounding 200,583 million into $200.6 billion invents a number the filing
    does not print, even when the row label is in the chunk.
    """
    for amount in amount_tokens(answer):
        if _chunk_has_amount(cited, amount):
            continue
        raise GroundingError(f"{amount} is not in the cited passage")


def _chunk_has_amount(cited: list[tuple[str, list[str], str]], amount: str) -> bool:
    wanted = _amount_key(amount)
    if not wanted:
        return False
    for _, _, content in cited:
        if any(_amount_key(token) == wanted for token in _TOKEN.findall(content)):
            return True
    return False


def _reject_missing_date(
    question: str, answer: str, cited: list[tuple[str, list[str], str]]
) -> None:
    """A month or weekday the model adds has to be in the cited chunk.

    A date the question already stated is the analyst's wording, not a new claim.
    """
    haystack = _chunk_text(cited).casefold()
    if not haystack:
        return
    asked = set(_date_words(question))
    for word in _date_words(answer):
        if word in asked or word in haystack:
            continue
        raise GroundingError(f"{word} is not in the cited passage")


def _chunk_text(cited: list[tuple[str, list[str], str]]) -> str:
    return "\n".join(content for _, _, content in cited if content.strip())


def _text_has_percent(text: str, percent: str) -> bool:
    wanted = _numeric_key(percent)
    if not wanted:
        return False
    if re.search(rf"(?<!\d){re.escape(wanted)}%", text):
        return True
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = _cells(line)
        number = any(_amount_key(cell) == wanted for cell in cells)
        mark = any(cell.strip() == "%" for cell in cells)
        if number and mark:
            return True
    return False


def _has_year_table(excerpt: str, context: list[str]) -> bool:
    return any(_header_for(line, context) is not None for line in _table_lines(excerpt))


def _columns_for(amount: str, cited: list[tuple[str, list[str], str]]) -> set[int]:
    found: set[int] = set()
    for excerpt, context, _content in cited:
        for line in _table_lines(excerpt):
            header = _header_for(line, context)
            if header is None or not _line_has_figure(line, amount):
                continue
            found |= _years_for_amount(header, line, amount)
    return found


def _saw_year_header(cited: list[tuple[str, list[str], str]]) -> bool:
    for excerpt, context, _content in cited:
        if _has_year_table(excerpt, context):
            return True
    return False


def _amount_year_pairs(text: str) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for match in _AMOUNT_THEN_YEAR.finditer(text):
        amount = match.group(1)
        year = int(match.group(2))
        pair = (amount, year)
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _percent_year_pairs(text: str) -> list[tuple[str, int]]:
    if re.search(r"\brespectively\b", text, re.IGNORECASE):
        percents = [_numeric_key(token) for token in _PERCENT.findall(text)]
        years = question_years(text)
        if percents and len(percents) == len(years):
            return list(zip(percents, years, strict=True))
    pairs: list[tuple[str, int]] = []
    found = list(_PERCENT.finditer(text))
    for index, match in enumerate(found):
        end = found[index + 1].start() if index + 1 < len(found) else len(text)
        year = _YEAR.search(text[match.end() : end])
        if year is None:
            continue
        pair = (_numeric_key(match.group(0)), int(year.group(1)))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _date_words(text: str) -> list[str]:
    lowered = text.casefold()
    words = [word for word in _WEEKDAYS if word in lowered]
    for word in _MONTHS:
        if word == "may":
            if re.search(r"\bMay\b", text):
                words.append(word)
            continue
        if word in lowered:
            words.append(word)
    return words


def year_column_lines(content: str, extra: list[str]) -> list[str]:
    """Label each table value with the year of its column.

    The header may be in this chunk or in a neighbor. The original row is
    unchanged; these lines are only a reading of it.
    """
    context = [content, *extra]
    labels: list[str] = []
    seen: set[str] = set()
    for line in _table_lines(content):
        if _is_year_header(line):
            continue
        header = _header_for(line, context)
        if header is None:
            continue
        cells = _cells(line)
        label = _row_label(cells)
        if not label:
            continue
        years: list[int] = []
        for year in header:
            if year is not None and year not in years:
                years.append(year)
        for year in years:
            value = _value_for_year(header, cells, year)
            if value is None:
                continue
            text = f"{label} {year}: {value}"
            if text in seen:
                continue
            seen.add(text)
            labels.append(text)
    return labels


def _is_year_header(line: str) -> bool:
    cells = [cell.strip() for cell in _cells(line) if cell.strip()]
    if not any(_year_cell(cell) is not None for cell in cells):
        return False
    return not any("," in cell or "." in cell for cell in cells)


def _row_label(cells: list[str]) -> str:
    for cell in cells:
        text = cell.strip()
        if not text or text in {"$", "%"}:
            continue
        if _year_cell(text) is not None or _amount_key(text):
            continue
        return text
    return ""


def _value_for_year(
    header: list[int | None], cells: list[str], year: int
) -> str | None:
    values = [
        cell.strip()
        for index, cell in enumerate(cells)
        if index < len(header)
        and header[index] == year
        and _amount_key(cell)
        and _year_cell(cell.strip()) is None
    ]
    if not values:
        return None
    amounts = [value for value in values if "," in value or "." in value]
    return amounts[0] if amounts else values[0]


def _table_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if line.strip().startswith("|") and "---" not in line
    ]


def _header_for(line: str, context: list[str]) -> list[int | None] | None:
    width = len(_cells(line))
    if width == 0:
        return None
    headers: list[list[int | None]] = []
    for block in context:
        for candidate in block.splitlines():
            parsed = _cells(candidate)
            if len(parsed) != width:
                continue
            years = [_year_cell(cell) for cell in parsed]
            if any(year is not None for year in years):
                headers.append(years)
    if not headers:
        return None
    return max(headers, key=lambda years: sum(year is not None for year in years))


def _years_for_amount(
    header: list[int | None], line: str, amount: str
) -> set[int]:
    wanted = _amount_key(amount)
    found: set[int] = set()
    for index, cell in enumerate(_cells(line)):
        if _amount_key(cell) != wanted:
            continue
        if index < len(header) and header[index] is not None:
            found.add(header[index])
    return found


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    parts = stripped.split("|")
    if len(parts) < 3:
        return []
    return [part.strip() for part in parts[1:-1]]


def _year_cell(cell: str) -> int | None:
    if _YEAR_CELL.fullmatch(cell.strip()):
        return int(cell.strip())
    return None


def _line_has_label(line: str, label: str) -> bool:
    if line.strip().startswith("|"):
        return any(_cell_has_label(cell, label) for cell in _cells(line))
    pattern = r"(?<![A-Za-z])" + re.escape(label) + r"(?![A-Za-z])"
    return re.search(pattern, line, re.IGNORECASE) is not None


def _cell_has_label(cell: str, label: str) -> bool:
    text = " ".join(cell.split()).casefold()
    wanted = " ".join(label.split()).casefold()
    if not text or not wanted:
        return False
    if text == wanted:
        return True
    if not text.startswith(wanted):
        return False
    return text[len(wanted) : len(wanted) + 1] in {" ", "("}


def _line_has_figure(line: str, figure: str) -> bool:
    wanted = _numeric_key(figure)
    if not wanted:
        return False
    if line.strip().startswith("|"):
        return any(_amount_key(cell) == wanted for cell in _cells(line))
    return any(_amount_key(token) == wanted for token in _TOKEN.findall(line))


def _numeric_key(value: str) -> str:
    match = _TOKEN.search(value)
    if match is None:
        return ""
    return _amount_key(match.group(0))


def _amount_key(value: str) -> str:
    cleaned = (
        value.strip().replace("$", "").replace(",", "").replace(" ", "").replace("%", "")
    )
    cleaned = cleaned.strip("()")
    if not re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return ""
    return cleaned
