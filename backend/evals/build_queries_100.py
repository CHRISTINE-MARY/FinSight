"""Build evals/queries_100.json from filing chunks.

Run from backend/:

    python evals/build_queries_100.py

These 100 questions do not repeat the 50 in ground_truth.json.
Each answerable excerpt is an exact slice of the cited chunk.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from sqlalchemy import text

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database.documents import database_engine

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "queries_100.json"
OLD = ROOT / "ground_truth.json"
REFUSAL = "The corpus does not contain enough evidence."
FIGURES = re.compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+\.\d+%?|\d+%|\b(?:19|20)\d{2}\b"
)

# ticker, fiscal year, phrase that must occur in that filing.
Src = tuple[str, int, str]


def Q(
    qid: str,
    category: str,
    question: str,
    answer: str,
    sources: tuple[Src, ...] = (),
    claims: tuple[str, ...] = (),
    absent: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "id": qid,
        "category": category,
        "question": question,
        "expected_answer": answer,
        "sources": sources,
        "claims": claims,
        "absent": absent,
    }


def questions() -> list[dict[str, object]]:
    return [
        Q(
            "qf-01",
            "factual",
            "Where does Apple's fiscal 2025 Form 10-K place its principal executive offices?",
            "One Apple Park Way.",
            (("AAPL", 2025, "One Apple Park Way"),),
            ("one apple park way",),
        ),
        Q(
            "qf-02",
            "factual",
            "Where does Microsoft's fiscal 2025 Form 10-K place its principal executive offices?",
            "One Microsoft Way, Redmond, Washington.",
            (("MSFT", 2025, "ONE MICROSOFT WAY, REDMOND"),),
            ("one microsoft way", "redmond"),
        ),
        Q(
            "qf-03",
            "factual",
            "Where does Amazon's fiscal 2025 Form 10-K place its principal executive offices?",
            "410 Terry Avenue North, Seattle, Washington.",
            (("AMZN", 2025, "410 Terry Avenue North"),),
            ("410 terry avenue", "seattle"),
        ),
        Q(
            "qf-04",
            "factual",
            "Where does Alphabet's fiscal 2025 Form 10-K place its principal executive offices?",
            "1600 Amphitheatre Parkway, Mountain View, California.",
            (("GOOGL", 2025, "1600 Amphitheatre Parkway"),),
            ("1600 amphitheatre", "mountain view"),
        ),
        Q(
            "qf-05",
            "factual",
            "Where does NVIDIA's fiscal 2025 Form 10-K place its principal executive offices?",
            "2788 San Tomas Expressway, Santa Clara, California.",
            (("NVDA", 2025, "2788 San Tomas Expressway"),),
            ("2788 san tomas", "santa clara"),
        ),
        Q(
            "qf-06",
            "factual",
            "Which firm has audited Apple, and since what year, in the fiscal 2025 Form 10-K?",
            "Ernst & Young LLP has served as Apple's auditor since 2009.",
            (("AAPL", 2025, "auditor since 2009"),),
            ("ernst & young", "2009"),
        ),
        Q(
            "qf-07",
            "factual",
            "Which independent firm audited Microsoft's fiscal 2025 consolidated financial statements?",
            "Deloitte & Touche LLP.",
            (("MSFT", 2025, "Deloitte & Touche LLP"),),
            ("deloitte & touche",),
        ),
        Q(
            "qf-08",
            "factual",
            "Which firm audited NVIDIA's internal control over financial reporting as of January 26, 2025?",
            "PricewaterhouseCoopers LLP.",
            (("NVDA", 2025, "audited by PricewaterhouseCoopers LLP"),),
            ("pricewaterhousecoopers",),
        ),
        Q(
            "qf-09",
            "factual",
            "What geographic reportable segments does Apple name in its fiscal 2025 Form 10-K?",
            "The Americas, Europe, Greater China, Japan, and Rest of Asia Pacific.",
            (("AAPL", 2025, "Greater China, Japan and Rest of Asia Pacific"),),
            ("greater china", "rest of asia pacific"),
        ),
        Q(
            "qf-10",
            "factual",
            "What three segments does Microsoft use to measure financial performance in fiscal 2025?",
            "Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.",
            (
                (
                    "MSFT",
                    2025,
                    "Productivity and Business Processes, Intelligent Cloud, and More Personal Computing",
                ),
            ),
            ("intelligent cloud", "more personal computing"),
        ),
        Q(
            "qf-11",
            "factual",
            "How does Alphabet's fiscal 2025 Form 10-K define YouTube ads revenue?",
            "YouTube ads includes revenues generated on YouTube properties.",
            (("GOOGL", 2025, "revenues generated on YouTube properties"),),
            ("youtube properties",),
        ),
        Q(
            "qf-12",
            "factual",
            "What does NVIDIA say its CUDA programming model runs on, in the fiscal 2025 Form 10-K?",
            "The foundational CUDA programming model runs on all NVIDIA GPUs.",
            (("NVDA", 2025, "foundational CUDA programming model that runs on all NVIDIA GPUs"),),
            ("cuda programming model",),
        ),
        Q(
            "qf-13",
            "factual",
            "In how many countries did NVIDIA have employees at the end of fiscal 2025?",
            "38 countries.",
            (("NVDA", 2025, "36,000 employees in 38 countries"),),
            ("38 countries",),
        ),
        Q(
            "qf-14",
            "factual",
            "What advertising programs does Amazon name for advertisers in its fiscal 2025 Form 10-K?",
            "Sponsored ads, display, and video advertising.",
            (("AMZN", 2025, "sponsored ads, display, and video advertising"),),
            ("sponsored ads",),
        ),
        Q(
            "qf-15",
            "factual",
            "How much commercial paper did Apple have outstanding on September 27, 2025?",
            "$8.0 billion, payable within 12 months.",
            (("AAPL", 2025, "$8.0 billion of commercial paper outstanding"),),
            ("8.0",),
        ),
        Q(
            "qf-16",
            "factual",
            "How much of Microsoft's $60.0 billion share repurchase program remained on June 30, 2025?",
            "$57.3 billion.",
            (("MSFT", 2025, "$57.3 billion remained of this $60.0 billion"),),
            ("57.3", "60.0"),
        ),
        Q(
            "qf-17",
            "factual",
            "How much cash dividends did NVIDIA pay shareholders in fiscal 2025?",
            "$834 million.",
            (("NVDA", 2025, "cash dividends to our shareholders of $834 million"),),
            ("834 million",),
        ),
        Q(
            "qf-18",
            "factual",
            "How much stock did NVIDIA repurchase in fiscal 2025, and for what amount?",
            "310 million shares for $34.0 billion.",
            (("NVDA", 2025, "repurchased 310 million shares of our common stock for $34.0 billion"),),
            ("310 million", "34.0"),
        ),
        Q(
            "qf-19",
            "factual",
            "What date ended Apple's fiscal 2025 year on the cover of its Form 10-K?",
            "September 27, 2025.",
            (("AAPL", 2025, "For the fiscal year ended September 27, 2025"),),
            ("september 27, 2025",),
        ),
        Q(
            "qf-20",
            "factual",
            "What date ended Microsoft's fiscal 2025 year on its Form 10-K index?",
            "June 30, 2025.",
            (("MSFT", 2025, "For the Fiscal Year Ended June 30, 2025"),),
            ("june 30, 2025",),
        ),
        Q(
            "qn-01",
            "numerical",
            "How many full-time equivalent employees did Apple have as of September 27, 2025?",
            "Approximately 166,000.",
            (("AAPL", 2025, "September 27, 2025, the Company had approximately 166,000"),),
        ),
        Q(
            "qn-02",
            "numerical",
            "How many full-time equivalent employees did Apple have in its fiscal 2023 Form 10-K?",
            "Approximately 161,000.",
            (("AAPL", 2023, "approximately 161,000 full-time equivalent employees"),),
        ),
        Q(
            "qn-03",
            "numerical",
            "How many full-time equivalent employees did Apple have in its fiscal 2022 Form 10-K?",
            "Approximately 164,000.",
            (("AAPL", 2022, "approximately 164,000 full-time equivalent employees"),),
        ),
        Q(
            "qn-04",
            "numerical",
            "How many employees did Alphabet have as of December 31, 2025?",
            "190,820.",
            (("GOOGL", 2025, "Alphabet had 190,820 employees"),),
        ),
        Q(
            "qn-05",
            "numerical",
            "How many employees did Alphabet have as of December 31, 2024?",
            "183,323.",
            (("GOOGL", 2024, "Alphabet had 183,323 employees"),),
        ),
        Q(
            "qn-06",
            "numerical",
            "How many employees did Alphabet have as of December 31, 2023?",
            "182,502.",
            (("GOOGL", 2023, "Alphabet had 182,502 employees"),),
        ),
        Q(
            "qn-07",
            "numerical",
            "How many employees did Alphabet have as of December 31, 2022?",
            "190,234.",
            (("GOOGL", 2022, "Alphabet had 190,234 employees"),),
        ),
        Q(
            "qn-08",
            "numerical",
            "How many full-time and part-time employees did Amazon have as of December 31, 2025?",
            "Approximately 1,576,000.",
            (("AMZN", 2025, "employed approximately 1,576,000 full-time and part-time"),),
        ),
        Q(
            "qn-09",
            "numerical",
            "How many full-time and part-time employees did Amazon have as of December 31, 2024?",
            "Approximately 1,556,000.",
            (("AMZN", 2024, "employed approximately 1,556,000 full-time and part-time"),),
        ),
        Q(
            "qn-10",
            "numerical",
            "How many full-time and part-time employees did Amazon have as of December 31, 2023?",
            "Approximately 1,525,000.",
            (("AMZN", 2023, "employed approximately 1,525,000 full-time and part-time"),),
        ),
        Q(
            "qn-11",
            "numerical",
            "How many people did Microsoft employ full-time as of June 30, 2025?",
            "Approximately 228,000.",
            (("MSFT", 2025, "As of June 30, 2025, we employed approximately 228,000"),),
        ),
        Q(
            "qn-12",
            "numerical",
            "How many of Microsoft's June 30, 2025 employees were in the United States?",
            "125,000.",
            (("MSFT", 2025, "228,000 people on a full-time basis, 125,000 in the U.S."),),
        ),
        Q(
            "qn-13",
            "numerical",
            "How many of Microsoft's June 30, 2025 employees were outside the United States?",
            "103,000.",
            (("MSFT", 2025, "125,000 in the U.S. and 103,000 internationally"),),
        ),
        Q(
            "qn-14",
            "numerical",
            "How many people did Microsoft employ full-time as of June 30, 2024?",
            "Approximately 228,000.",
            (("MSFT", 2024, "As of June 30, 2024, we employed approximately 228,000"),),
        ),
        Q(
            "qn-15",
            "numerical",
            "How many of Microsoft's June 30, 2024 employees were in the United States?",
            "126,000.",
            (("MSFT", 2024, "228,000 people on a full-time basis, 126,000 in the U.S."),),
        ),
        Q(
            "qn-16",
            "numerical",
            "How many of Microsoft's June 30, 2024 employees worked outside the United States?",
            "102,000.",
            (("MSFT", 2024, "126,000 in the U.S. and 102,000 internationally"),),
        ),
        Q(
            "qn-17",
            "numerical",
            "How many employees did NVIDIA have at the end of fiscal 2025?",
            "Approximately 36,000.",
            (("NVDA", 2025, "approximately 36,000 employees in 38 countries"),),
        ),
        Q(
            "qn-18",
            "numerical",
            "How many of NVIDIA's fiscal 2025 employees worked in research and development?",
            "27,100.",
            (("NVDA", 2025, "27,100 were engaged in research and development"),),
        ),
        Q(
            "qn-19",
            "numerical",
            "How many of NVIDIA's fiscal 2025 employees worked in sales, marketing, operations, and administration?",
            "8,900.",
            (("NVDA", 2025, "8,900 were engaged in sales, marketing, operations"),),
        ),
        Q(
            "qn-20",
            "numerical",
            "How many employees did NVIDIA have at the end of fiscal 2024?",
            "Approximately 29,600.",
            (("NVDA", 2024, "approximately 29,600 employees in 36 countries"),),
        ),
        Q(
            "qn-21",
            "numerical",
            "How many Microsoft employees were in operations as of June 30, 2025?",
            "89,000.",
            (("MSFT", 2025, "89,000 were in operations"),),
        ),
        Q(
            "qn-22",
            "numerical",
            "How many Microsoft employees were in operations as of June 30, 2024?",
            "86,000.",
            (("MSFT", 2024, "86,000 were in operations"),),
        ),
        Q(
            "qn-23",
            "numerical",
            "What diluted earnings per share did Microsoft report for the year in the column printed as 13.64?",
            "13.64.",
            (("MSFT", 2025, "13.64"),),
        ),
        Q(
            "qn-24",
            "numerical",
            "What total revenue amount is printed beside Total revenue in NVIDIA's fiscal 2025 Form 10-K?",
            "130,497.",
            (("NVDA", 2025, "130,497"),),
        ),
        Q(
            "qh-01",
            "multi_hop",
            "How many full-time equivalent employees did Apple have on September 27, 2025, and where are its principal executive offices?",
            "Approximately 166,000 employees, at One Apple Park Way.",
            (
                ("AAPL", 2025, "September 27, 2025, the Company had approximately 166,000"),
                ("AAPL", 2025, "One Apple Park Way"),
            ),
            ("one apple park way",),
        ),
        Q(
            "qh-02",
            "multi_hop",
            "How many people did Microsoft employ on June 30, 2025, and where are its principal executive offices?",
            "Approximately 228,000 people, at One Microsoft Way in Redmond, Washington.",
            (
                ("MSFT", 2025, "As of June 30, 2025, we employed approximately 228,000"),
                ("MSFT", 2025, "ONE MICROSOFT WAY, REDMOND"),
            ),
            ("one microsoft way",),
        ),
        Q(
            "qh-03",
            "multi_hop",
            "How many employees did Amazon have on December 31, 2025, and where are its principal executive offices?",
            "Approximately 1,576,000 employees, at 410 Terry Avenue North in Seattle.",
            (
                ("AMZN", 2025, "employed approximately 1,576,000 full-time and part-time"),
                ("AMZN", 2025, "410 Terry Avenue North"),
            ),
            ("410 terry avenue",),
        ),
        Q(
            "qh-04",
            "multi_hop",
            "How many employees did Alphabet have on December 31, 2025, and where are its principal executive offices?",
            "190,820 employees, at 1600 Amphitheatre Parkway in Mountain View.",
            (
                ("GOOGL", 2025, "Alphabet had 190,820 employees"),
                ("GOOGL", 2025, "1600 Amphitheatre Parkway"),
            ),
            ("1600 amphitheatre",),
        ),
        Q(
            "qh-05",
            "multi_hop",
            "How many employees did NVIDIA have at the end of fiscal 2025, and where are its principal executive offices?",
            "Approximately 36,000 employees, at 2788 San Tomas Expressway in Santa Clara.",
            (
                ("NVDA", 2025, "approximately 36,000 employees in 38 countries"),
                ("NVDA", 2025, "2788 San Tomas Expressway"),
            ),
            ("2788 san tomas",),
        ),
        Q(
            "qh-06",
            "multi_hop",
            "Who has audited Apple since 2009, and how much commercial paper was outstanding on September 27, 2025?",
            "Ernst & Young LLP, and $8.0 billion of commercial paper.",
            (
                ("AAPL", 2025, "auditor since 2009"),
                ("AAPL", 2025, "$8.0 billion of commercial paper outstanding"),
            ),
            ("ernst & young",),
        ),
        Q(
            "qh-07",
            "multi_hop",
            "Who audited Microsoft's fiscal 2025 financial statements, and which three segments does it report?",
            "Deloitte & Touche LLP. The segments are Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.",
            (
                ("MSFT", 2025, "Deloitte & Touche LLP"),
                (
                    "MSFT",
                    2025,
                    "Productivity and Business Processes, Intelligent Cloud, and More Personal Computing",
                ),
            ),
            ("deloitte & touche", "intelligent cloud"),
        ),
        Q(
            "qh-08",
            "multi_hop",
            "Who audited NVIDIA as of January 26, 2025, and how much in cash dividends did NVIDIA pay in fiscal 2025?",
            "PricewaterhouseCoopers LLP, and $834 million of cash dividends.",
            (
                ("NVDA", 2025, "audited by PricewaterhouseCoopers LLP"),
                ("NVDA", 2025, "cash dividends to our shareholders of $834 million"),
            ),
            ("pricewaterhousecoopers", "834 million"),
        ),
        Q(
            "qh-09",
            "multi_hop",
            "What were Apple's and Microsoft's full-time headcounts on their fiscal 2025 year-ends?",
            "Apple had approximately 166,000 full-time equivalent employees and Microsoft employed approximately 228,000 people.",
            (
                ("AAPL", 2025, "approximately 166,000 full-time equivalent employees"),
                ("MSFT", 2025, "we employed approximately 228,000"),
            ),
        ),
        Q(
            "qh-10",
            "multi_hop",
            "What were Alphabet's and Amazon's headcounts at December 31, 2025?",
            "Alphabet had 190,820 employees and Amazon employed approximately 1,576,000.",
            (
                ("GOOGL", 2025, "Alphabet had 190,820 employees"),
                ("AMZN", 2025, "employed approximately 1,576,000"),
            ),
        ),
        Q(
            "qh-11",
            "multi_hop",
            "How many employees did NVIDIA have at the end of fiscal 2024 and at the end of fiscal 2025?",
            "Approximately 29,600 at the end of fiscal 2024 and approximately 36,000 at the end of fiscal 2025.",
            (
                ("NVDA", 2024, "approximately 29,600 employees"),
                ("NVDA", 2025, "approximately 36,000 employees"),
            ),
        ),
        Q(
            "qh-12",
            "multi_hop",
            "What geographic segments does Apple report, and how many full-time equivalent employees did it have on September 27, 2025?",
            "Americas, Europe, Greater China, Japan, and Rest of Asia Pacific, and approximately 166,000 employees.",
            (
                ("AAPL", 2025, "Greater China, Japan and Rest of Asia Pacific"),
                ("AAPL", 2025, "approximately 166,000 full-time equivalent employees"),
            ),
            ("greater china",),
        ),
        Q(
            "qh-13",
            "multi_hop",
            "How does Alphabet define YouTube ads, and how many employees did it have on December 31, 2025?",
            "YouTube ads are revenues generated on YouTube properties. Alphabet had 190,820 employees.",
            (
                ("GOOGL", 2025, "revenues generated on YouTube properties"),
                ("GOOGL", 2025, "Alphabet had 190,820 employees"),
            ),
            ("youtube properties",),
        ),
        Q(
            "qh-14",
            "multi_hop",
            "What advertising formats does Amazon name, and how many employees did it have on December 31, 2025?",
            "Sponsored ads, display, and video advertising. Amazon employed approximately 1,576,000 people.",
            (
                ("AMZN", 2025, "sponsored ads, display, and video advertising"),
                ("AMZN", 2025, "employed approximately 1,576,000"),
            ),
            ("sponsored ads",),
        ),
        Q(
            "qh-15",
            "multi_hop",
            "How much of Microsoft's repurchase authorization remained on June 30, 2025, and how many people did it employ that day?",
            "$57.3 billion remained of the $60.0 billion program, and Microsoft employed approximately 228,000 people.",
            (
                ("MSFT", 2025, "$57.3 billion remained of this $60.0 billion"),
                ("MSFT", 2025, "we employed approximately 228,000"),
            ),
        ),
        Q(
            "qh-16",
            "multi_hop",
            "How many shares did NVIDIA repurchase in fiscal 2025, and how many employees did it have at year-end?",
            "310 million shares for $34.0 billion, and approximately 36,000 employees.",
            (
                ("NVDA", 2025, "repurchased 310 million shares of our common stock for $34.0 billion"),
                ("NVDA", 2025, "approximately 36,000 employees"),
            ),
            ("34.0",),
        ),
        Q(
            "qh-17",
            "multi_hop",
            "On what date did Apple's fiscal 2025 year end, and how many full-time equivalent employees did it have that day?",
            "September 27, 2025, with approximately 166,000 full-time equivalent employees.",
            (
                ("AAPL", 2025, "For the fiscal year ended September 27, 2025"),
                ("AAPL", 2025, "September 27, 2025, the Company had approximately 166,000"),
            ),
            ("september 27, 2025",),
        ),
        Q(
            "qh-18",
            "multi_hop",
            "On what date did Microsoft's fiscal 2025 year end, and how many people did it employ that day?",
            "June 30, 2025, with approximately 228,000 full-time employees.",
            (
                ("MSFT", 2025, "For the Fiscal Year Ended June 30, 2025"),
                ("MSFT", 2025, "As of June 30, 2025, we employed approximately 228,000"),
            ),
            ("june 30, 2025",),
        ),
        Q(
            "qh-19",
            "multi_hop",
            "How many NVIDIA employees worked in research and development, and how many worked in sales, marketing, operations, and administration, at the end of fiscal 2025?",
            "27,100 in research and development and 8,900 in sales, marketing, operations, and administration.",
            (
                ("NVDA", 2025, "27,100 were engaged in research and development"),
                ("NVDA", 2025, "8,900 were engaged in sales, marketing, operations"),
            ),
        ),
        Q(
            "qh-20",
            "multi_hop",
            "How many U.S. and international employees did Microsoft report on June 30, 2025?",
            "125,000 in the United States and 103,000 internationally.",
            (
                ("MSFT", 2025, "125,000 in the U.S. and 103,000 internationally"),
                ("MSFT", 2025, "As of June 30, 2025, we employed approximately 228,000"),
            ),
        ),
        Q(
            "qt-01",
            "temporal",
            "What full-time equivalent employee counts did Apple report for fiscal 2022, fiscal 2023, and fiscal 2025, in that order?",
            "164,000, then 161,000, then 166,000.",
            (
                ("AAPL", 2022, "approximately 164,000 full-time equivalent employees"),
                ("AAPL", 2023, "approximately 161,000 full-time equivalent employees"),
                ("AAPL", 2025, "approximately 166,000 full-time equivalent employees"),
            ),
        ),
        Q(
            "qt-02",
            "temporal",
            "What employee counts did Alphabet report at December 31, 2022, 2023, 2024, and 2025, in that order?",
            "190,234, then 182,502, then 183,323, then 190,820.",
            (
                ("GOOGL", 2022, "Alphabet had 190,234 employees"),
                ("GOOGL", 2023, "Alphabet had 182,502 employees"),
                ("GOOGL", 2024, "Alphabet had 183,323 employees"),
                ("GOOGL", 2025, "Alphabet had 190,820 employees"),
            ),
        ),
        Q(
            "qt-03",
            "temporal",
            "What employee counts did Amazon report at December 31, 2023, 2024, and 2025, in that order?",
            "1,525,000, then 1,556,000, then 1,576,000.",
            (
                ("AMZN", 2023, "employed approximately 1,525,000"),
                ("AMZN", 2024, "employed approximately 1,556,000"),
                ("AMZN", 2025, "employed approximately 1,576,000"),
            ),
        ),
        Q(
            "qt-04",
            "temporal",
            "What employee counts did NVIDIA report at the end of fiscal 2024 and fiscal 2025, in that order?",
            "29,600, then 36,000.",
            (
                ("NVDA", 2024, "approximately 29,600 employees"),
                ("NVDA", 2025, "approximately 36,000 employees"),
            ),
        ),
        Q(
            "qt-05",
            "temporal",
            "How many people did Microsoft employ full-time on June 30, 2024 and on June 30, 2025, in that order?",
            "228,000, then 228,000.",
            (
                ("MSFT", 2024, "As of June 30, 2024, we employed approximately 228,000"),
                ("MSFT", 2025, "As of June 30, 2025, we employed approximately 228,000"),
            ),
        ),
        Q(
            "qt-06",
            "temporal",
            "How many of Microsoft's employees were in the United States on June 30, 2024 and on June 30, 2025, in that order?",
            "126,000, then 125,000.",
            (
                ("MSFT", 2024, "126,000 in the U.S."),
                ("MSFT", 2025, "125,000 in the U.S."),
            ),
        ),
        Q(
            "qt-07",
            "temporal",
            "How many of Microsoft's employees were outside the United States on June 30, 2024 and on June 30, 2025, in that order?",
            "102,000, then 103,000.",
            (
                ("MSFT", 2024, "102,000 internationally"),
                ("MSFT", 2025, "103,000 internationally"),
            ),
        ),
        Q(
            "qt-08",
            "temporal",
            "How many Microsoft employees were in operations on June 30, 2024 and on June 30, 2025, in that order?",
            "86,000, then 89,000.",
            (
                ("MSFT", 2024, "86,000 were in operations"),
                ("MSFT", 2025, "89,000 were in operations"),
            ),
        ),
        Q(
            "qt-09",
            "temporal",
            "Did Alphabet's employee count rise or fall from December 31, 2024 to December 31, 2025, and what were the two counts?",
            "It rose, from 183,323 to 190,820.",
            (
                ("GOOGL", 2024, "Alphabet had 183,323 employees"),
                ("GOOGL", 2025, "Alphabet had 190,820 employees"),
            ),
        ),
        Q(
            "qt-10",
            "temporal",
            "Did Amazon's employee count rise or fall from December 31, 2023 to December 31, 2025, and what were the two counts?",
            "It rose, from 1,525,000 to 1,576,000.",
            (
                ("AMZN", 2023, "employed approximately 1,525,000"),
                ("AMZN", 2025, "employed approximately 1,576,000"),
            ),
        ),
        Q(
            "qt-11",
            "temporal",
            "Did NVIDIA's employee count rise or fall from the end of fiscal 2024 to the end of fiscal 2025, and what were the two counts?",
            "It rose, from 29,600 to 36,000.",
            (
                ("NVDA", 2024, "approximately 29,600 employees"),
                ("NVDA", 2025, "approximately 36,000 employees"),
            ),
        ),
        Q(
            "qt-12",
            "temporal",
            "Did Apple's full-time equivalent employee count rise or fall from fiscal 2023 to fiscal 2025, and what were the two counts?",
            "It rose, from 161,000 to 166,000.",
            (
                ("AAPL", 2023, "approximately 161,000 full-time equivalent employees"),
                ("AAPL", 2025, "approximately 166,000 full-time equivalent employees"),
            ),
        ),
        Q(
            "qt-13",
            "temporal",
            "Did Alphabet's employee count rise or fall from December 31, 2022 to December 31, 2023, and what were the two counts?",
            "It fell, from 190,234 to 182,502.",
            (
                ("GOOGL", 2022, "Alphabet had 190,234 employees"),
                ("GOOGL", 2023, "Alphabet had 182,502 employees"),
            ),
        ),
        Q(
            "qt-14",
            "temporal",
            "Which year-end headcount was higher for Alphabet, December 31, 2022 or December 31, 2025?",
            "December 31, 2025 was higher, at 190,820 versus 190,234.",
            (
                ("GOOGL", 2022, "Alphabet had 190,234 employees"),
                ("GOOGL", 2025, "Alphabet had 190,820 employees"),
            ),
        ),
        Q(
            "qt-15",
            "temporal",
            "Which Apple full-time equivalent count was higher, fiscal 2022 or fiscal 2023?",
            "Fiscal 2022 was higher, at 164,000 versus 161,000.",
            (
                ("AAPL", 2022, "approximately 164,000 full-time equivalent employees"),
                ("AAPL", 2023, "approximately 161,000 full-time equivalent employees"),
            ),
        ),
        Q(
            "qt-16",
            "temporal",
            "In how many countries did NVIDIA report employees at the end of fiscal 2024 and fiscal 2025, in that order?",
            "36 countries, then 38 countries.",
            (
                ("NVDA", 2024, "29,600 employees in 36 countries"),
                ("NVDA", 2025, "36,000 employees in 38 countries"),
            ),
            ("36 countries", "38 countries"),
        ),
        Q(
            "qt-17",
            "temporal",
            "What were Amazon's employee counts at December 31, 2024 and December 31, 2025, in that order?",
            "1,556,000, then 1,576,000.",
            (
                ("AMZN", 2024, "employed approximately 1,556,000"),
                ("AMZN", 2025, "employed approximately 1,576,000"),
            ),
        ),
        Q(
            "qt-18",
            "temporal",
            "What were Microsoft's U.S. employee counts on June 30, 2025 and its international count that same day?",
            "125,000 in the U.S. and 103,000 internationally.",
            (("MSFT", 2025, "125,000 in the U.S. and 103,000 internationally"),),
        ),
        Q(
            "qt-19",
            "temporal",
            "What were Alphabet's employee counts at December 31, 2023 and December 31, 2024, in that order?",
            "182,502, then 183,323.",
            (
                ("GOOGL", 2023, "Alphabet had 182,502 employees"),
                ("GOOGL", 2024, "Alphabet had 183,323 employees"),
            ),
        ),
        Q(
            "qt-20",
            "temporal",
            "What were Apple's full-time equivalent counts for fiscal 2023 and fiscal 2022, in that order?",
            "161,000, then 164,000.",
            (
                ("AAPL", 2023, "approximately 161,000 full-time equivalent employees"),
                ("AAPL", 2022, "approximately 164,000 full-time equivalent employees"),
            ),
        ),
        Q(
            "qu-01",
            "unanswerable",
            "What was Meta Platforms' fiscal 2024 revenue?",
            REFUSAL,
            absent=("Meta Platforms",),
        ),
        Q(
            "qu-02",
            "unanswerable",
            "How many paid memberships did Netflix report at the end of 2024?",
            REFUSAL,
            absent=("Netflix",),
        ),
        Q(
            "qu-03",
            "unanswerable",
            "What was Intel's foundry revenue for 2024?",
            REFUSAL,
            absent=("Intel",),
        ),
        Q(
            "qu-04",
            "unanswerable",
            "What was AMD's data center revenue for 2024?",
            REFUSAL,
            absent=("Advanced Micro Devices",),
        ),
        Q(
            "qu-05",
            "unanswerable",
            "What was Oracle's cloud revenue for fiscal 2024?",
            REFUSAL,
            absent=("Oracle",),
        ),
        Q(
            "qu-06",
            "unanswerable",
            "What was Salesforce's subscription revenue for fiscal 2025?",
            REFUSAL,
            absent=("Salesforce",),
        ),
        Q(
            "qu-07",
            "unanswerable",
            "What were Uber's gross bookings for 2024?",
            REFUSAL,
            absent=("Uber",),
        ),
        Q(
            "qu-08",
            "unanswerable",
            "What was Disney's parks operating income for fiscal 2024?",
            REFUSAL,
            absent=("Disney",),
        ),
        Q(
            "qu-09",
            "unanswerable",
            "What was JPMorgan Chase's net interest income for 2024?",
            REFUSAL,
            absent=("JPMorgan",),
        ),
        Q(
            "qu-10",
            "unanswerable",
            "What unit case volume did Coca-Cola report for 2024?",
            REFUSAL,
            absent=("Coca-Cola",),
        ),
        Q(
            "qu-11",
            "unanswerable",
            "What was Pfizer's COVID-19 vaccine revenue for 2024?",
            REFUSAL,
            absent=("Pfizer",),
        ),
        Q(
            "qu-12",
            "unanswerable",
            "How many commercial airplanes did Boeing deliver in 2024?",
            REFUSAL,
            absent=("Boeing",),
        ),
        Q(
            "qu-13",
            "unanswerable",
            "How many electric vehicles did Ford Motor sell in 2024?",
            REFUSAL,
            absent=("Ford Motor",),
        ),
        Q(
            "qu-14",
            "unanswerable",
            "What was Samsung Electronics' semiconductor revenue for 2024?",
            REFUSAL,
            absent=("Samsung",),
        ),
        Q(
            "qu-15",
            "unanswerable",
            "What was Adobe's Creative Cloud revenue for fiscal 2024?",
            REFUSAL,
            absent=("Adobe",),
        ),
        Q(
            "qu-16",
            "unanswerable",
            "How many barrels of oil did Chevron produce in 2024?",
            REFUSAL,
            absent=("Chevron",),
        ),
    ]


def find_chunk(conn, ticker: str, fiscal_year: int, phrase: str) -> tuple[str, str, str, str]:
    row = conn.execute(
        text(
            """
            SELECT c.id::text, c.content, d.company_name, d.filing_type
            FROM document_chunks c
            JOIN source_documents d ON d.id = c.document_id
            WHERE d.ticker = :ticker
              AND d.fiscal_year = :year
              AND position(lower(:phrase) in lower(c.content)) > 0
            ORDER BY length(c.content), c.chunk_index
            LIMIT 1
            """
        ),
        {"ticker": ticker, "year": fiscal_year, "phrase": phrase},
    ).first()
    if row is None:
        raise SystemExit(f"missing {ticker} FY{fiscal_year}: {phrase!r}")
    return str(row[0]), str(row[1]), str(row[2]), str(row[3])


def excerpt_for(content: str, phrase: str) -> str:
    start = content.lower().find(phrase.lower())
    if start < 0:
        raise SystemExit(f"phrase disappeared: {phrase!r}")
    end = min(len(content), start + max(len(phrase), 240))
    snippet = content[start:end].split("\n", 1)[0].strip()
    if phrase.lower() not in snippet.lower():
        snippet = content[start : start + len(phrase)]
    return snippet


def check_figures(answer: str, contents: list[str], years: set[int]) -> None:
    sources = contents + [str(year) for year in years]
    for figure in FIGURES.findall(answer):
        if not any(figure in source for source in sources):
            raise SystemExit(f"figure {figure} is not in a cited chunk: {answer}")


def check_absent(conn, name: str) -> None:
    found = conn.execute(
        text(
            """
            SELECT count(*) FROM source_documents
            WHERE company_name ILIKE :name OR ticker ILIKE :name
            """
        ),
        {"name": f"%{name}%"},
    ).scalar_one()
    if int(found) != 0:
        raise SystemExit(f"{name} is in the corpus")


def build() -> None:
    bank = json.loads(OLD.read_text(encoding="utf-8"))
    old_questions = {item["question"] for item in bank["questions"]}
    rows = questions()
    if len(rows) != 100:
        raise SystemExit(f"expected 100 questions, found {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate question id")
    engine = database_engine()
    built: list[dict[str, object]] = []
    with engine.connect() as conn:
        for row in rows:
            question = str(row["question"])
            if question in old_questions:
                raise SystemExit(f"repeats the original set: {question}")
            sources = row["sources"]
            evidence: list[dict[str, object]] = []
            contents: list[str] = []
            years: set[int] = set()
            seen: set[str] = set()
            for ticker, year, phrase in sources:  # type: ignore[misc]
                chunk_id, content, company, filing = find_chunk(conn, ticker, year, phrase)
                if filing != "10-K":
                    raise SystemExit(f"{chunk_id} is a {filing}")
                years.add(year)
                contents.append(content)
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                evidence.append(
                    {
                        "chunk_id": chunk_id,
                        "ticker": ticker,
                        "company_name": company,
                        "fiscal_year": year,
                        "filing_type": filing,
                        "excerpt": excerpt_for(content, phrase),
                    }
                )
            answer = str(row["expected_answer"])
            if row["category"] == "unanswerable":
                if answer != REFUSAL or sources:
                    raise SystemExit(f"{row['id']} must refuse with no chunks")
                for name in row["absent"]:  # type: ignore[union-attr]
                    check_absent(conn, str(name))
            else:
                if not evidence:
                    raise SystemExit(f"{row['id']} has no source")
                check_figures(answer, contents, years)
                for claim in row["claims"]:  # type: ignore[union-attr]
                    if not any(str(claim) in content.lower() for content in contents):
                        raise SystemExit(f"{row['id']} claim {claim!r} is not in a chunk")
            record: dict[str, object] = {
                "id": row["id"],
                "category": row["category"],
                "question": question,
                "expected_answer": answer,
                "chunk_ids": [item["chunk_id"] for item in evidence],
                "evidence": evidence,
            }
            if row["claims"]:
                record["claims"] = list(row["claims"])  # type: ignore[arg-type]
            built.append(record)
    counts: dict[str, int] = {}
    for record in built:
        counts[str(record["category"])] = counts.get(str(record["category"]), 0) + 1
    payload = {
        "description": (
            "A second 100-question set. None of these questions are in ground_truth.json. "
            "Answerable excerpts are exact slices of document_chunks. "
            "A correct answer must contain every figure in expected_answer and every claim."
        ),
        "refusal": REFUSAL,
        "counts": counts,
        "questions": built,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name} {counts}")


if __name__ == "__main__":
    build()
