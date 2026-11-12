"""Load data/markdown filings into public.source_documents.

Run from backend/ so app settings and the database driver are available:

    uv run python ../data/load_source_documents.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine.url import make_url

from app.config import settings
from app.database.models.source_document import SourceDocument

DATA_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = DATA_DIR / "downloads" / "manifest.json"
MARKDOWN_DIR = DATA_DIR / "markdown"

COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
}


def _database_url(url: str) -> str:
    parsed = make_url(url)
    if parsed.drivername != "postgresql+psycopg":
        parsed = parsed.set(drivername="postgresql+psycopg")
    return parsed.render_as_string(hide_password=False)


def _fiscal_year(filing: dict[str, str]) -> int:
    report_date = filing.get("report_date") or ""
    filing_date = filing["filing_date"]
    return int((report_date or filing_date)[:4])


def _rows() -> list[dict[str, object]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for filing in payload["filings"]:
        ticker = str(filing["ticker"])
        company_name = COMPANY_NAMES.get(ticker)
        if company_name is None:
            raise RuntimeError(f"No company name mapped for {ticker}")
        markdown_path = MARKDOWN_DIR / Path(str(filing["local_path"])).with_suffix(".md")
        if not markdown_path.is_file():
            raise FileNotFoundError(markdown_path)
        report_date = filing.get("report_date") or None
        rows.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "cik": str(filing["cik"]),
                "filing_type": str(filing["form"]),
                "filing_date": date.fromisoformat(str(filing["filing_date"])),
                "report_date": date.fromisoformat(report_date) if report_date else None,
                "fiscal_year": _fiscal_year(filing),
                "accession_number": str(filing["accession_number"]),
                "source_url": str(filing["source_url"]),
                "markdown": markdown_path.read_text(encoding="utf-8"),
            }
        )
    return rows


def load_source_documents() -> int:
    rows = _rows()
    engine = create_engine(_database_url(settings.database_url), pool_pre_ping=True)
    statement = insert(SourceDocument).values(rows)
    excluded = statement.excluded
    statement = statement.on_conflict_do_update(
        constraint="uq_source_documents_accession_number",
        set_={
            "ticker": excluded.ticker,
            "company_name": excluded.company_name,
            "cik": excluded.cik,
            "filing_type": excluded.filing_type,
            "filing_date": excluded.filing_date,
            "report_date": excluded.report_date,
            "fiscal_year": excluded.fiscal_year,
            "source_url": excluded.source_url,
            "markdown": excluded.markdown,
            "updated_at": func.now(),
        },
    )
    with engine.begin() as connection:
        connection.execute(statement)
        stored = connection.execute(
            select(
                SourceDocument.ticker,
                SourceDocument.fiscal_year,
                SourceDocument.accession_number,
                func.length(SourceDocument.markdown),
            )
            .where(
                SourceDocument.accession_number.in_(
                    [str(row["accession_number"]) for row in rows]
                )
            )
            .order_by(SourceDocument.ticker, SourceDocument.fiscal_year)
        ).all()
    for ticker, fiscal_year, accession_number, characters in stored:
        print(f"{ticker} FY{fiscal_year} {accession_number} {characters} chars")
    return len(stored)


if __name__ == "__main__":
    count = load_source_documents()
    print(f"Stored {count} source document(s)")
