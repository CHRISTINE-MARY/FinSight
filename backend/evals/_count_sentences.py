"""Count narrative sentences that can anchor eval questions."""

import re

from sqlalchemy import text

from app.database.documents import database_engine

FIGURES = re.compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\$\d|\d+\.\d+%?|\d+%|\b(?:19|20)\d{2}\b"
)
SKIP = re.compile(
    r"check mark|forward-looking|table of contents|incorporated by reference|"
    r"securities exchange act|form 10-k|signature|page \d+|item \d",
    re.I,
)


def sentences(content: str) -> list[str]:
    if content.count("|") > 4:
        return []
    parts = re.split(r"(?<=[.])\s+", content)
    kept: list[str] = []
    for part in parts:
        part = " ".join(part.split())
        if not 90 <= len(part) <= 380:
            continue
        if "|" in part or SKIP.search(part):
            continue
        if not FIGURES.search(part):
            continue
        if sum(ch.isalpha() for ch in part) < 40:
            continue
        kept.append(part)
    return kept


def main() -> None:
    engine = database_engine()
    seen: set[str] = set()
    by_key: dict[tuple[str, int], int] = {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT d.ticker, d.fiscal_year, c.content
                FROM document_chunks c
                JOIN source_documents d ON d.id = c.document_id
                WHERE d.filing_type = '10-K'
                """
            )
        ).all()
    for ticker, year, content in rows:
        for sentence in sentences(content):
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            by_key[(ticker, year)] = by_key.get((ticker, year), 0) + 1
    print("unique", len(seen))
    for key in sorted(by_key):
        print(key, by_key[key])


if __name__ == "__main__":
    main()
