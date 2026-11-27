"""Print a few chunk shapes so the query builder can target them."""

import sys

from sqlalchemy import text

from app.database.documents import database_engine


def main() -> None:
    engine = database_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT d.ticker, d.fiscal_year, left(c.content, 700)
                FROM document_chunks c
                JOIN source_documents d ON d.id = c.document_id
                WHERE d.ticker = 'AAPL' AND d.fiscal_year = 2024
                  AND c.content ~ '[$][0-9]'
                ORDER BY c.chunk_index
                LIMIT 6
                """
            )
        ).all()
    for ticker, year, content in rows:
        print("=" * 20, ticker, year)
        print(content)
        print()


if __name__ == "__main__":
    sys.exit(main())
