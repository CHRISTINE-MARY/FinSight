"""One-off corpus inventory for the query-set builder."""

from sqlalchemy import text

from app.database.documents import database_engine


def main() -> None:
    engine = database_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT d.ticker, d.company_name, d.fiscal_year, count(*)
                FROM source_documents d
                JOIN document_chunks c ON c.document_id = d.id
                WHERE d.filing_type = '10-K'
                GROUP BY 1, 2, 3
                ORDER BY 1, 3
                """
            )
        ).all()
        print("filings", len(rows))
        for row in rows:
            print(row[0], row[2], row[3], row[1])
        print("chunks", conn.execute(text("SELECT count(*) FROM document_chunks")).scalar())


if __name__ == "__main__":
    main()
