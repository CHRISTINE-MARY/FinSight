"""Build evals/ground_truth.json from document_chunks.

Run from backend/:

    python evals/build_ground_truth.py

Answerable excerpts are read back from the cited chunk and must occur there.
Unanswerable questions stay empty unless the absence checks fail.
This does not call the assistant.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database.documents import database_engine

REFUSAL = "The corpus does not contain enough evidence."
OUT = Path(__file__).with_name("ground_truth.json")
FIGURES = re.compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+\.\d+%?|\d+%|\b(?:19|20)\d{2}\b"
)


@dataclass(frozen=True)
class Span:
    chunk_id: str
    ticker: str
    fiscal_year: int
    phrases: tuple[str, ...]
    max_span: int = 500


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    question: str
    expected_answer: str
    spans: tuple[Span, ...] = ()
    order: tuple[tuple[str, tuple[str, ...]], ...] = ()
    absence: tuple[tuple[str, int], ...] = ()
    note: str = ""


def cover(content: str, phrases: tuple[str, ...], max_span: int) -> str:
    """Return the shortest exact slice that contains phrases in order."""
    start = 0
    first = phrases[0]
    while True:
        pos = content.find(first, start)
        if pos < 0:
            raise SystemExit(f"missing phrase: {first!r}")
        window = content[pos : pos + max_span]
        cursor = 0
        matched = True
        for phrase in phrases:
            found = window.find(phrase, cursor)
            if found < 0:
                matched = False
                break
            cursor = found + len(phrase)
        if matched:
            return content[pos : pos + cursor]
        start = pos + 1


def assert_order(content: str, phrases: tuple[str, ...], chunk_id: str) -> None:
    pos = -1
    for phrase in phrases:
        found = content.find(phrase, pos + 1)
        if found < 0:
            raise SystemExit(f"{chunk_id} lost order at {phrase!r}")
        pos = found


def filing(conn, chunk_id: str) -> tuple[str, int, str, str]:
    row = conn.execute(
        text(
            """
            SELECT d.ticker, d.fiscal_year, d.company_name, d.filing_type
            FROM document_chunks c
            JOIN source_documents d ON d.id = c.document_id
            WHERE c.id = CAST(:id AS uuid)
            """
        ),
        {"id": chunk_id},
    ).one()
    return str(row[0]), int(row[1]), str(row[2]), str(row[3])


def load_chunk(conn, chunk_id: str) -> str:
    return conn.execute(
        text("SELECT content FROM document_chunks WHERE id = CAST(:id AS uuid)"),
        {"id": chunk_id},
    ).scalar_one()


def check_figures(answer: str, contents: list[str], years: set[int]) -> None:
    sources = contents + [str(year) for year in years]
    for figure in FIGURES.findall(answer):
        if not any(figure in source for source in sources):
            raise SystemExit(f"figure {figure} is not in a cited chunk: {answer}")


def build_item(conn, item: Item) -> dict[str, object]:
    evidence: list[dict[str, object]] = []
    contents: list[str] = []
    seen: dict[str, str] = {}
    for span in item.spans:
        if span.chunk_id not in seen:
            seen[span.chunk_id] = load_chunk(conn, span.chunk_id)
        content = seen[span.chunk_id]
        ticker, fiscal_year, company_name, filing_type = filing(conn, span.chunk_id)
        if ticker != span.ticker or fiscal_year != span.fiscal_year:
            raise SystemExit(
                f"{item.id} {span.chunk_id} is {ticker} FY{fiscal_year}, "
                f"expected {span.ticker} FY{span.fiscal_year}"
            )
        if filing_type != "10-K":
            raise SystemExit(f"{span.chunk_id} is a {filing_type}")
        excerpt = cover(content, span.phrases, span.max_span)
        evidence.append(
            {
                "chunk_id": span.chunk_id,
                "ticker": ticker,
                "company_name": company_name,
                "fiscal_year": fiscal_year,
                "filing_type": filing_type,
                "excerpt": excerpt,
            }
        )
        contents.append(content)
    for chunk_id, phrases in item.order:
        assert_order(seen[chunk_id], phrases, chunk_id)
    if item.category == "unanswerable":
        if item.expected_answer != REFUSAL or item.spans:
            raise SystemExit(f"{item.id} must refuse with no chunks")
        for sql, expect in item.absence:
            found = int(conn.execute(text(sql)).scalar_one())
            if found != expect:
                raise SystemExit(f"{item.id} absence {sql!r} returned {found}")
    else:
        if not item.spans:
            raise SystemExit(f"{item.id} has no source chunk")
        check_figures(
            item.expected_answer,
            contents,
            {span.fiscal_year for span in item.spans},
        )
    record: dict[str, object] = {
        "id": item.id,
        "category": item.category,
        "question": item.question,
        "expected_answer": item.expected_answer,
        "chunk_ids": list(dict.fromkeys(span.chunk_id for span in item.spans)),
        "evidence": evidence,
    }
    if item.note:
        record["note"] = item.note
    return record


def span(
    chunk_id: str,
    ticker: str,
    fiscal_year: int,
    *phrases: str,
    max_span: int = 500,
) -> Span:
    return Span(chunk_id, ticker, fiscal_year, phrases, max_span)


AAPL_FY = "The Company's fiscal year is the 52- or 53-week period that ends on the last Saturday of September."
AAPL_HEADCOUNT = (
    "As of September 28, 2024, the Company had approximately "
    "164,000 full-time equivalent employees."
)
CAT_2024 = "net sales by category for 2024, 2023 and 2022 (dollars in millions):"
CAT_2023 = "net sales by category for 2023, 2022 and 2021 (dollars in millions):"
CAT_2022 = "net sales by category for 2022, 2021 and 2020 (dollars in millions):"
CAT_2021 = "net sales by category for 2021, 2020 and 2019 (dollars in millions):"
SEG_2024 = "net sales by reportable segment for 2024, 2023 and 2022 (dollars in millions):"
SEG_2025 = "net sales by reportable segment for 2025, 2024 and 2023 (dollars in millions):"
GOOGL_SEGMENTS = (
    "We report Google in two segments, Google Services and Google Cloud, "
    "and all non-Google businesses collectively as Other Bets."
)
AMZN_SEGMENTS = (
    "three segments: North America, International, and Amazon Web Services"
)
GPU = "Our invention of the GPU in 1999 sparked the growth of the PC gaming market"
RND = "We have invested over $58.2 billion in research and development since our inception"
IPHONE_FLAT = "iPhone net sales were relatively flat during 2024 compared to 2023."
MAC_UP = "Mac net sales increased during 2024 compared to 2023"
CUSTOMERS = (
    "Sales to direct Customers A, B and C represented 12%, 11% and 11% of "
    "total revenue, respectively, for fiscal year 2025, all of which were "
    "primarily attributable to the Compute & Networking segment."
)
OUTSIDE_US = (
    "Revenue from sales to customers outside of the United States accounted "
    "for 53% and 56% of total revenue for fiscal years 2025 and 2024, respectively."
)
FY2023_RESULTS = (
    "The Company's total net sales were $383.3 billion and net income was "
    "$97.0 billion during 2023."
)
FY2023_DOWN = (
    "The Company's total net sales decreased 3% or $11.0 billion during 2023 "
    "compared to 2022."
)
FY2022_UP = "Total net sales increased 8% or $28.5 billion during 2022 compared to 2021"
FY2021_UP = "Total net sales increased 33% or $91.3 billion during 2021 compared to 2020"
NVDA_2023 = "fiscal year 2023 revenue was $26.97 billion, flat compared with a year ago."
GOOGL_2022 = "As of December 31, 2022, we had 190,234 employees."
GOOGL_2023 = "As of December 31, 2023, we had 182,502 employees."
GOOGL_2024 = "As of December 31, 2024, we had 183,323 employees."
GOOGL_2025 = "As of December 31, 2025, we had 190,820 employees."
MSFT_UNIT = "(In millions, except per share amounts)"


def items() -> list[Item]:
    iphone_2024 = "d4417312-73b6-4791-825f-88d34ff19b63"
    services_2024 = "8d86e5ec-67f3-42b1-acaf-b768e77cd098"
    cat_2024 = "c472acdb-c6d1-4cdb-9d54-c1c9591f0cab"
    total_2024 = "6c86071f-b043-4fa6-a748-87d12bc64251"
    seg_2024 = "895d2ddf-a1af-4f9f-bf3d-13d35fc288f5"
    headcount = "efdcb657-8f4e-498f-90a4-fbee5ef881ec"
    total_2025 = "8970deaf-425c-4c61-b298-2fb619f4765b"
    seg_2025 = "e1e4886d-dd0f-4ad9-b4b3-5e08d496b404"
    iphone_2023 = "98cee2bc-8324-48b6-bc4d-f51e32ea1ff3"
    cat_2023 = "7cccddba-e0b7-4a5d-9364-f7837a8c5e90"
    iphone_2022 = "dee0073d-3597-44dc-ad1d-3204892f729e"
    cat_2022 = "33407953-3503-4516-b3b5-94a9de5ac63f"
    iphone_2021 = "4bd915a4-f32a-460f-bdee-f512ac76f276"
    cat_2021 = "f121a068-dbbe-4fca-8f72-2fca77a00ad6"
    services_2022 = "f4fb624b-0bc3-4142-81c7-c718f8b63419"
    services_2021 = "f4e95ac2-c252-4b39-a905-d6f0a87d49cf"
    msft = "d8a88a49-fe25-424b-8138-5abbdb380ade"
    return [
        Item(
            "fact-01",
            "factual",
            "When does Apple's fiscal year end?",
            AAPL_FY,
            (span(seg_2024, "AAPL", 2024, AAPL_FY),),
        ),
        Item(
            "fact-02",
            "factual",
            "How does Alphabet organize its reported segments in its fiscal 2024 Form 10-K?",
            "Alphabet reports Google in two segments, Google Services and Google Cloud, "
            "and all non-Google businesses collectively as Other Bets.",
            (
                span(
                    "bf5d10b1-3a34-4c48-a4a9-fed675b558e3",
                    "GOOGL",
                    2024,
                    GOOGL_SEGMENTS,
                ),
            ),
        ),
        Item(
            "fact-03",
            "factual",
            "Which operating segments does Amazon report in its fiscal 2024 Form 10-K?",
            "North America, International, and Amazon Web Services (AWS).",
            (
                span(
                    "512650b8-8739-431d-883d-b20c8207225a",
                    "AMZN",
                    2024,
                    AMZN_SEGMENTS,
                ),
            ),
        ),
        Item(
            "fact-04",
            "factual",
            "In which year does NVIDIA say it invented the GPU, in its fiscal 2025 Form 10-K?",
            "1999.",
            (span("1e422cc5-e952-420d-8d3a-9b666f0e40e2", "NVDA", 2025, GPU),),
        ),
        Item(
            "fact-05",
            "factual",
            "How much does NVIDIA say it has invested in research and development since "
            "its inception, in its fiscal 2025 Form 10-K?",
            "Over $58.2 billion.",
            (span("1e422cc5-e952-420d-8d3a-9b666f0e40e2", "NVDA", 2025, RND),),
        ),
        Item(
            "fact-06",
            "factual",
            "How did Apple describe the change in iPhone net sales from fiscal 2023 to fiscal 2024?",
            IPHONE_FLAT,
            (span(services_2024, "AAPL", 2024, IPHONE_FLAT),),
        ),
        Item(
            "fact-07",
            "factual",
            "Did Apple's Mac net sales increase or decrease in fiscal 2024 compared with fiscal 2023?",
            "Mac net sales increased during 2024 compared to 2023.",
            (span(services_2024, "AAPL", 2024, MAC_UP),),
        ),
        Item(
            "fact-08",
            "factual",
            "What share of NVIDIA's fiscal 2025 total revenue came from its largest direct customers?",
            "Direct Customers A, B and C represented 12%, 11% and 11% of total revenue, "
            "respectively, all primarily attributable to the Compute & Networking segment.",
            (span("6c7caab9-a981-49c4-a4e5-eb6e9bed6b2b", "NVDA", 2025, CUSTOMERS),),
        ),
        Item(
            "fact-09",
            "factual",
            "What share of NVIDIA revenue came from customers outside the United States "
            "in fiscal 2025 and fiscal 2024?",
            "53% for fiscal 2025 and 56% for fiscal 2024.",
            (span("3be6c625-e9c1-4b61-a477-cff1ec9f495c", "NVDA", 2025, OUTSIDE_US),),
        ),
        Item(
            "fact-10",
            "factual",
            "What were Apple's total net sales and net income for fiscal 2023?",
            "Total net sales were $383.3 billion and net income was $97.0 billion.",
            (span("8e5d0263-fbfc-4a93-a426-0cf421a5ef98", "AAPL", 2023, FY2023_RESULTS),),
        ),
        Item(
            "num-01",
            "numerical",
            "How many full-time equivalent employees did Apple have as of September 28, 2024?",
            "Approximately 164,000.",
            (span(headcount, "AAPL", 2024, AAPL_HEADCOUNT),),
        ),
        Item(
            "num-02",
            "numerical",
            "What were Apple's iPhone net sales for fiscal 2024?",
            "The fiscal 2024 category table prints iPhone net sales of 201,183, in dollars in millions.",
            (
                span(iphone_2024, "AAPL", 2024, "| iPhone", "201,183", max_span=180),
                span(cat_2024, "AAPL", 2024, CAT_2024),
            ),
        ),
        Item(
            "num-03",
            "numerical",
            "What were Apple's Mac net sales for fiscal 2024?",
            "The fiscal 2024 category table prints Mac net sales of 29,984, in dollars in millions.",
            (
                span(iphone_2024, "AAPL", 2024, "| Mac", "29,984", max_span=160),
                span(cat_2024, "AAPL", 2024, CAT_2024),
            ),
        ),
        Item(
            "num-04",
            "numerical",
            "What were Apple's iPad net sales for fiscal 2024?",
            "The fiscal 2024 category table prints iPad net sales of 26,694, in dollars in millions.",
            (
                span(iphone_2024, "AAPL", 2024, "| iPad", "26,694", max_span=160),
                span(cat_2024, "AAPL", 2024, CAT_2024),
            ),
        ),
        Item(
            "num-05",
            "numerical",
            "What were Apple's Services net sales for fiscal 2024?",
            "The fiscal 2024 category table prints Services net sales of 96,169, in dollars in millions.",
            (
                span(services_2024, "AAPL", 2024, "| Services (1)", "96,169", max_span=160),
                span(cat_2024, "AAPL", 2024, CAT_2024),
            ),
        ),
        Item(
            "num-06",
            "numerical",
            "What were Apple's Wearables, Home and Accessories net sales for fiscal 2024?",
            "The fiscal 2024 category table prints Wearables, Home and Accessories net sales "
            "of 37,005, in dollars in millions.",
            (
                span(
                    services_2024,
                    "AAPL",
                    2024,
                    "| Wearables, Home and Accessories",
                    "37,005",
                    max_span=180,
                ),
                span(cat_2024, "AAPL", 2024, CAT_2024),
            ),
        ),
        Item(
            "num-07",
            "numerical",
            "What were Apple's total net sales for fiscal 2024?",
            "The fiscal 2024 reportable-segment table prints total net sales of 391,035, "
            "in dollars in millions.",
            (
                span(total_2024, "AAPL", 2024, "| Total net sales", "391,035", max_span=180),
                span(seg_2024, "AAPL", 2024, SEG_2024),
            ),
        ),
        Item(
            "num-08",
            "numerical",
            "What were Apple's total net sales for fiscal 2025?",
            "The fiscal 2025 reportable-segment table prints total net sales of 416,161, "
            "in dollars in millions.",
            (
                span(total_2025, "AAPL", 2025, "| Total net sales", "416,161", max_span=180),
                span(seg_2025, "AAPL", 2025, SEG_2025),
            ),
        ),
        Item(
            "num-09",
            "numerical",
            "What were Apple's iPhone net sales for fiscal 2023?",
            "The fiscal 2023 category table prints iPhone net sales of 200,583, in dollars in millions.",
            (
                span(iphone_2023, "AAPL", 2023, "iPhone (1)", "200,583", max_span=140),
                span(cat_2023, "AAPL", 2023, CAT_2023),
            ),
        ),
        Item(
            "num-10",
            "numerical",
            "What were Apple's iPhone net sales for fiscal 2022?",
            "The fiscal 2022 category table prints iPhone net sales of 205,489, in dollars in millions.",
            (
                span(iphone_2022, "AAPL", 2022, "iPhone (1)", "205,489", max_span=140),
                span(cat_2022, "AAPL", 2022, CAT_2022),
            ),
        ),
        Item(
            "num-11",
            "numerical",
            "What were Apple's iPhone net sales for fiscal 2021?",
            "The fiscal 2021 category table prints iPhone net sales of 191,973, in dollars in millions.",
            (
                span(iphone_2021, "AAPL", 2021, "iPhone (1)", "191,973", max_span=140),
                span(cat_2021, "AAPL", 2021, CAT_2021),
            ),
        ),
        Item(
            "num-12",
            "numerical",
            "What did Microsoft report as Product revenue for the year ended June 30, 2024?",
            "The income statement prints Product revenue of 64,773. Amounts are in millions, "
            "except per share amounts.",
            (
                span(msft, "MSFT", 2024, "Product", "64,773", max_span=120),
                span(msft, "MSFT", 2024, MSFT_UNIT),
                span(msft, "MSFT", 2024, "Year Ended June 30", "2024", max_span=80),
            ),
            order=((msft, ("Product", "64,773", "Service and other")),),
        ),
        Item(
            "hop-01",
            "multi_hop",
            "In Apple's fiscal 2024 category table, which was larger, iPhone or Services net sales, "
            "and what were the two amounts?",
            "iPhone net sales were 201,183 and Services net sales were 96,169, in dollars in millions. "
            "iPhone was larger.",
            (
                span(iphone_2024, "AAPL", 2024, "| iPhone", "201,183", max_span=180),
                span(services_2024, "AAPL", 2024, "| Services (1)", "96,169", max_span=160),
                span(cat_2024, "AAPL", 2024, CAT_2024),
            ),
        ),
        Item(
            "hop-02",
            "multi_hop",
            "In Apple's fiscal 2024 category table, what were Mac net sales and "
            "Wearables, Home and Accessories net sales?",
            "Mac net sales were 29,984 and Wearables, Home and Accessories net sales were 37,005, "
            "in dollars in millions.",
            (
                span(iphone_2024, "AAPL", 2024, "| Mac", "29,984", max_span=160),
                span(
                    services_2024,
                    "AAPL",
                    2024,
                    "| Wearables, Home and Accessories",
                    "37,005",
                    max_span=180,
                ),
                span(cat_2024, "AAPL", 2024, CAT_2024),
            ),
        ),
        Item(
            "hop-03",
            "multi_hop",
            "What were Apple's fiscal 2024 total net sales and Services net sales?",
            "Reportable-segment total net sales were 391,035 and category Services net sales "
            "were 96,169, both in dollars in millions.",
            (
                span(total_2024, "AAPL", 2024, "| Total net sales", "391,035", max_span=180),
                span(seg_2024, "AAPL", 2024, SEG_2024),
                span(services_2024, "AAPL", 2024, "| Services (1)", "96,169", max_span=160),
                span(cat_2024, "AAPL", 2024, CAT_2024),
            ),
        ),
        Item(
            "hop-04",
            "multi_hop",
            "How many full-time equivalent employees did Apple have as of September 28, 2024, "
            "and when does its fiscal year end?",
            "Approximately 164,000. The fiscal year is the 52- or 53-week period that ends "
            "on the last Saturday of September.",
            (
                span(headcount, "AAPL", 2024, AAPL_HEADCOUNT),
                span(seg_2024, "AAPL", 2024, AAPL_FY),
            ),
        ),
        Item(
            "hop-05",
            "multi_hop",
            "For NVIDIA's fiscal 2025, what share of revenue came from direct Customers A, B and C, "
            "and what share came from customers outside the United States in fiscal 2025 and fiscal 2024?",
            "Customers A, B and C represented 12%, 11% and 11% of fiscal 2025 total revenue. "
            "Customers outside the United States accounted for 53% of fiscal 2025 revenue "
            "and 56% of fiscal 2024 revenue.",
            (
                span("6c7caab9-a981-49c4-a4e5-eb6e9bed6b2b", "NVDA", 2025, CUSTOMERS),
                span("3be6c625-e9c1-4b61-a477-cff1ec9f495c", "NVDA", 2025, OUTSIDE_US),
            ),
        ),
        Item(
            "hop-06",
            "multi_hop",
            "How much has NVIDIA invested in research and development since its inception, "
            "and what was its fiscal 2023 revenue?",
            "Its fiscal 2025 Form 10-K says it has invested over $58.2 billion since its inception. "
            "Its fiscal 2023 Form 10-K says fiscal year 2023 revenue was $26.97 billion.",
            (
                span("1e422cc5-e952-420d-8d3a-9b666f0e40e2", "NVDA", 2025, RND),
                span("040389e5-f14c-45bf-8b66-36898b9a4535", "NVDA", 2023, NVDA_2023),
            ),
        ),
        Item(
            "hop-07",
            "multi_hop",
            "How does Alphabet's fiscal 2024 Form 10-K organize its segments, "
            "and how many employees did it have as of December 31, 2024?",
            "It reports Google Services, Google Cloud, and Other Bets. "
            "As of December 31, 2024, it had 183,323 employees.",
            (
                span(
                    "bf5d10b1-3a34-4c48-a4a9-fed675b558e3",
                    "GOOGL",
                    2024,
                    GOOGL_SEGMENTS,
                ),
                span("71afc67b-7fba-4672-9052-0df5d5c31f14", "GOOGL", 2024, GOOGL_2024),
            ),
        ),
        Item(
            "hop-08",
            "multi_hop",
            "By how much did Apple's total net sales change in fiscal 2021, "
            "and what did that year's category table print for iPhone net sales?",
            "Total net sales increased 33% or $91.3 billion during 2021 compared to 2020. "
            "The category table prints iPhone net sales of 191,973, in dollars in millions.",
            (
                span(cat_2021, "AAPL", 2021, FY2021_UP),
                span(cat_2021, "AAPL", 2021, CAT_2021),
                span(iphone_2021, "AAPL", 2021, "iPhone (1)", "191,973", max_span=140),
            ),
        ),
        Item(
            "hop-09",
            "multi_hop",
            "How many employees did Alphabet report as of December 31, 2024, "
            "and how many full-time equivalent employees did Apple report as of September 28, 2024?",
            "Alphabet had 183,323 employees. Apple had approximately 164,000 full-time "
            "equivalent employees.",
            (
                span("71afc67b-7fba-4672-9052-0df5d5c31f14", "GOOGL", 2024, GOOGL_2024),
                span(headcount, "AAPL", 2024, AAPL_HEADCOUNT),
            ),
        ),
        Item(
            "hop-10",
            "multi_hop",
            "What were Apple's fiscal 2024 total net sales, and how many full-time equivalent "
            "employees did it have as of September 28, 2024?",
            "Reportable-segment total net sales were 391,035, in dollars in millions. "
            "Apple had approximately 164,000 full-time equivalent employees.",
            (
                span(total_2024, "AAPL", 2024, "| Total net sales", "391,035", max_span=180),
                span(seg_2024, "AAPL", 2024, SEG_2024),
                span(headcount, "AAPL", 2024, AAPL_HEADCOUNT),
            ),
        ),
        Item(
            "time-01",
            "temporal",
            "What iPhone net sales does Apple's fiscal 2024 category table print for 2024, 2023, and 2022?",
            "In dollars in millions, iPhone net sales are 201,183 for 2024, 200,583 for 2023, "
            "and 205,489 for 2022.",
            (
                span(
                    iphone_2024,
                    "AAPL",
                    2024,
                    "| iPhone",
                    "201,183",
                    "200,583",
                    "205,489",
                    max_span=320,
                ),
                span(cat_2024, "AAPL", 2024, CAT_2024),
            ),
            order=((iphone_2024, ("| iPhone", "201,183", "200,583", "205,489", "| Mac")),),
        ),
        Item(
            "time-02",
            "temporal",
            "What Services net sales does Apple's fiscal 2024 category table print for 2024, 2023, and 2022?",
            "In dollars in millions, Services net sales are 96,169 for 2024, 85,200 for 2023, "
            "and 78,129 for 2022.",
            (
                span(
                    services_2024,
                    "AAPL",
                    2024,
                    "| Services (1)",
                    "96,169",
                    "85,200",
                    "78,129",
                    max_span=350,
                ),
                span(cat_2024, "AAPL", 2024, CAT_2024),
            ),
            order=(
                (
                    services_2024,
                    ("| Services (1)", "96,169", "85,200", "78,129"),
                ),
            ),
        ),
        Item(
            "time-03",
            "temporal",
            "What Mac net sales does Apple's fiscal 2024 category table print for 2024, 2023, and 2022?",
            "In dollars in millions, Mac net sales are 29,984 for 2024, 29,357 for 2023, "
            "and 40,177 for 2022.",
            (
                span(
                    iphone_2024,
                    "AAPL",
                    2024,
                    "| Mac",
                    "29,984",
                    "29,357",
                    "40,177",
                    max_span=350,
                ),
                span(cat_2024, "AAPL", 2024, CAT_2024),
            ),
            order=((iphone_2024, ("| Mac", "29,984", "29,357", "40,177", "| iPad")),),
        ),
        Item(
            "time-04",
            "temporal",
            "How did Apple describe the change in total net sales for fiscal 2021, fiscal 2022, "
            "and fiscal 2023?",
            "Fiscal 2021 increased 33% or $91.3 billion compared with 2020. "
            "Fiscal 2022 increased 8% or $28.5 billion compared with 2021. "
            "Fiscal 2023 decreased 3% or $11.0 billion compared with 2022.",
            (
                span(cat_2021, "AAPL", 2021, FY2021_UP),
                span("9b8f30e7-e555-4797-a6eb-6395892f051b", "AAPL", 2022, FY2022_UP),
                span("8e5d0263-fbfc-4a93-a426-0cf421a5ef98", "AAPL", 2023, FY2023_DOWN),
            ),
        ),
        Item(
            "time-05",
            "temporal",
            "How many employees did Alphabet report at each December 31 year-end from 2022 through 2025?",
            "190,234 as of December 31, 2022; 182,502 as of December 31, 2023; "
            "183,323 as of December 31, 2024; and 190,820 as of December 31, 2025.",
            (
                span("10193e4a-9ca0-4ade-bbe2-69c6301385b3", "GOOGL", 2022, GOOGL_2022),
                span("da4788f0-36ef-4070-9810-a97e707aecd5", "GOOGL", 2023, GOOGL_2023),
                span("71afc67b-7fba-4672-9052-0df5d5c31f14", "GOOGL", 2024, GOOGL_2024),
                span("86de0caa-2842-4711-b811-6fb61c97f626", "GOOGL", 2025, GOOGL_2025),
            ),
        ),
        Item(
            "time-06",
            "temporal",
            "What total revenue did Microsoft report for the years ended June 30, 2024, 2023, and 2022?",
            "In millions, total revenue was 245,122 for 2024, 211,915 for 2023, and 198,270 for 2022.",
            (
                span(
                    msft,
                    "MSFT",
                    2024,
                    "Total revenue",
                    "245,122",
                    "211,915",
                    "198,270",
                    max_span=220,
                ),
                span(msft, "MSFT", 2024, MSFT_UNIT),
                span(
                    msft,
                    "MSFT",
                    2024,
                    "Year Ended June 30",
                    "2024",
                    "2023",
                    "2022",
                    max_span=120,
                ),
            ),
            order=(
                (
                    msft,
                    (
                        "Year Ended June 30",
                        "2024",
                        "2023",
                        "2022",
                        "Total revenue",
                        "245,122",
                        "211,915",
                        "198,270",
                    ),
                ),
            ),
        ),
        Item(
            "time-07",
            "temporal",
            "What iPhone net sales does each of Apple's fiscal 2021, 2022, 2023, and 2024 "
            "category tables print for that fiscal year?",
            "In dollars in millions, the current-year column is 191,973 for fiscal 2021, "
            "205,489 for fiscal 2022, 200,583 for fiscal 2023, and 201,183 for fiscal 2024.",
            (
                span(iphone_2021, "AAPL", 2021, "iPhone (1)", "191,973", max_span=140),
                span(cat_2021, "AAPL", 2021, CAT_2021),
                span(iphone_2022, "AAPL", 2022, "iPhone (1)", "205,489", max_span=140),
                span(cat_2022, "AAPL", 2022, CAT_2022),
                span(iphone_2023, "AAPL", 2023, "iPhone (1)", "200,583", max_span=140),
                span(cat_2023, "AAPL", 2023, CAT_2023),
                span(iphone_2024, "AAPL", 2024, "| iPhone", "201,183", max_span=180),
                span(cat_2024, "AAPL", 2024, CAT_2024),
            ),
        ),
        Item(
            "time-08",
            "temporal",
            "How did the total net sales printed in Apple's fiscal 2025 segment table compare "
            "with the total printed in the fiscal 2024 segment table?",
            "Fiscal 2025 prints 416,161 and fiscal 2024 prints 391,035, both in dollars in millions. "
            "The fiscal 2025 amount is higher.",
            (
                span(total_2025, "AAPL", 2025, "| Total net sales", "416,161", max_span=180),
                span(seg_2025, "AAPL", 2025, SEG_2025),
                span(total_2024, "AAPL", 2024, "| Total net sales", "391,035", max_span=180),
                span(seg_2024, "AAPL", 2024, SEG_2024),
            ),
        ),
        Item(
            "time-09",
            "temporal",
            "Did the share of NVIDIA revenue from customers outside the United States rise or fall "
            "from fiscal 2024 to fiscal 2025?",
            "It fell, from 56% for fiscal 2024 to 53% for fiscal 2025.",
            (span("3be6c625-e9c1-4b61-a477-cff1ec9f495c", "NVDA", 2025, OUTSIDE_US),),
        ),
        Item(
            "time-10",
            "temporal",
            "What Services net sales did Apple's fiscal 2021 and fiscal 2022 category tables print, "
            "and which year was higher?",
            "Fiscal 2021 prints 68,425 and fiscal 2022 prints 78,129, in dollars in millions. "
            "Fiscal 2022 was higher.",
            (
                span(services_2021, "AAPL", 2021, "Services (3)", "68,425", max_span=160),
                span(cat_2021, "AAPL", 2021, CAT_2021),
                span(services_2022, "AAPL", 2022, "Services (3)", "78,129", max_span=160),
                span(cat_2022, "AAPL", 2022, CAT_2022),
            ),
        ),
        Item(
            "gap-01",
            "unanswerable",
            "What was Tesla, Inc.'s automotive revenue for fiscal 2024?",
            REFUSAL,
            absence=(
                ("SELECT count(*) FROM source_documents WHERE ticker = 'TSLA'", 0),
                (
                    (
                        "SELECT count(*) FROM document_chunks "
                        "WHERE content ILIKE '%tesla%' "
                        "AND content ILIKE '%automotive revenue%'"
                    ),
                    0,
                ),
            ),
            note=(
                "NVIDIA filings mention Tesla, Inc. as an example. "
                "They do not report Tesla's automotive revenue."
            ),
        ),
        Item(
            "gap-02",
            "unanswerable",
            "What were Walmart's fiscal 2024 net sales?",
            REFUSAL,
            absence=(
                ("SELECT count(*) FROM source_documents WHERE ticker = 'WMT'", 0),
                ("SELECT count(*) FROM document_chunks WHERE content ILIKE '%walmart%'", 0),
            ),
        ),
        Item(
            "gap-03",
            "unanswerable",
            "What were Berkshire Hathaway's 2024 operating earnings?",
            REFUSAL,
            absence=(
                (
                    "SELECT count(*) FROM document_chunks WHERE content ILIKE '%berkshire%'",
                    0,
                ),
            ),
        ),
        Item(
            "gap-04",
            "unanswerable",
            "How much bitcoin did any company in the corpus hold?",
            REFUSAL,
            absence=(
                ("SELECT count(*) FROM document_chunks WHERE content ILIKE '%bitcoin%'", 0),
            ),
        ),
        Item(
            "gap-05",
            "unanswerable",
            "What was Shopify's fiscal 2024 revenue?",
            REFUSAL,
            absence=(
                ("SELECT count(*) FROM document_chunks WHERE content ILIKE '%shopify%'", 0),
            ),
        ),
        Item(
            "gap-06",
            "unanswerable",
            "What were ExxonMobil's 2024 upstream earnings?",
            REFUSAL,
            absence=(
                ("SELECT count(*) FROM document_chunks WHERE content ILIKE '%exxon%'", 0),
            ),
        ),
        Item(
            "gap-07",
            "unanswerable",
            "What were Costco's fiscal 2024 net sales?",
            REFUSAL,
            absence=(
                ("SELECT count(*) FROM document_chunks WHERE content ILIKE '%costco%'", 0),
            ),
        ),
        Item(
            "gap-08",
            "unanswerable",
            "What was Palantir's fiscal 2024 revenue?",
            REFUSAL,
            absence=(
                ("SELECT count(*) FROM document_chunks WHERE content ILIKE '%palantir%'", 0),
            ),
        ),
    ]


def main() -> None:
    bank = items()
    ids = [item.id for item in bank]
    if len(bank) != 50 or len(set(ids)) != 50:
        raise SystemExit(f"expected 50 unique questions, found {len(set(ids))}")
    counts: dict[str, int] = {}
    for item in bank:
        counts[item.category] = counts.get(item.category, 0) + 1
    engine = database_engine()
    with engine.connect() as conn:
        records = [build_item(conn, item) for item in bank]
    payload = {
        "description": (
            "Ground truth read from document_chunks before any assistant experiment. "
            "Every evidence excerpt is an exact substring of that chunk."
        ),
        "refusal": REFUSAL,
        "counts": counts,
        "questions": records,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name} {counts}")


if __name__ == "__main__":
    main()
