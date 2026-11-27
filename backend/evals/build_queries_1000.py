"""Build evals/queries_1000.json from filing chunks.

Run from backend/:

    python evals/build_queries_1000.py

Answerable questions are written from narrative sentences in the 10-K
chunks, then checked against those sentences. None of the questions repeat
ground_truth.json or queries_100.json. A checkpoint file lets a stopped run
continue. This does not call the assistant.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from sqlalchemy import text

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import NVIDIA_BASE_URL, settings
from app.database.documents import database_engine
from app.retrieval.scope import question_scopes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "queries_1000.json"
CHECKPOINT = ROOT / "queries_1000_checkpoint.jsonl"
OLD_FILES = (ROOT / "ground_truth.json", ROOT / "queries_100.json")
REFUSAL = "The corpus does not contain enough evidence."

FIGURES = re.compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+\.\d+%?|\d+%|\b(?:19|20)\d{2}\b"
)
VALUES = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+\.\d+%?|\d+%")
SENTENCE = re.compile(r"[^.|\n]{90,380}\.")
SKIP = re.compile(
    r"check mark|forward-looking|table of contents|incorporated by reference|"
    r"securities exchange act|exchange act|commission file number|"
    r"aggregate market value|see note|item 1a|item 8\b",
    re.IGNORECASE,
)
TOPICS = (
    "net sales",
    "total revenue",
    "revenue",
    "operating income",
    "net income",
    "gross margin",
    "earnings per share",
    "research and development",
    "repurchas",
    "dividend",
    "employees",
    "income tax",
    "effective tax",
    "cash and cash equivalents",
    "long-term debt",
    "operating expenses",
    "cost of sales",
    "cost of revenue",
)
WEAK_CLAIMS = {
    "the company",
    "fiscal year",
    "form 10-k",
    "million",
    "billion",
    "percent",
    "percentage",
}
TARGETS = {
    "factual": 200,
    "numerical": 240,
    "multi_hop": 200,
    "temporal": 200,
    "unanswerable": 160,
}
PREFIX = {
    "factual": "qf",
    "numerical": "qn",
    "multi_hop": "qh",
    "temporal": "qt",
    "unanswerable": "qu",
}
WORKERS = 4
BATCH = 6
# Names that are not filers in this corpus. Each must be absent from the chunks.
OUTSIDERS = (
    "Meta Platforms",
    "Netflix",
    "Intel",
    "Advanced Micro Devices",
    "Oracle",
    "Salesforce",
    "Uber",
    "Walt Disney",
    "JPMorgan Chase",
    "Coca-Cola",
    "Pfizer",
    "Boeing",
    "Ford Motor",
    "Samsung Electronics",
    "Adobe",
    "Chevron",
    "Tesla",
    "Walmart",
    "Exxon Mobil",
    "Bank of America",
    "Visa Inc",
    "Mastercard",
    "Home Depot",
    "Procter & Gamble",
    "Cisco",
    "Accenture",
    "Broadcom",
    "Costco",
    "AbbVie",
    "Nike",
    "PepsiCo",
    "Honeywell",
    "UnitedHealth",
    "Caterpillar",
    "Qualcomm",
    "Starbucks",
    "Lockheed Martin",
    "BlackRock",
    "Goldman Sachs",
    "Morgan Stanley",
    "Comcast",
    "AT&T",
    "Verizon",
    "IBM",
    "ServiceNow",
    "Shopify",
    "PayPal",
    "Block Inc",
    "Snap Inc",
    "Pinterest",
    "Spotify",
    "Airbnb",
    "Palantir",
    "Snowflake",
    "CrowdStrike",
    "Workday",
    "Intuit",
    "Texas Instruments",
    "Applied Materials",
)
METRIC_TEMPLATES = (
    "What was {name}'s total revenue for fiscal {year}?",
    "What was {name}'s net income for fiscal {year}?",
    "How many employees did {name} report at the end of fiscal {year}?",
    "Which firm audited {name}'s fiscal {year} financial statements?",
    "What diluted earnings per share did {name} report for fiscal {year}?",
    "How much did {name} spend on research and development in fiscal {year}?",
    "What was {name}'s operating income for fiscal {year}?",
    "How much cash did {name} report at the end of fiscal {year}?",
)
SYSTEM = """You write evaluation questions about SEC Form 10-K filings.
Use only the passages given. Copy every digit, comma, decimal, dollar sign, and percent sign exactly as printed.
Return one JSON object: {"items":[{"id":"","question":"","expected_answer":"","claims":[]}]}
Rules:
- Name the company in the question.
- Refer to each year only as "fiscal YYYY". Do not write month names or calendar dates.
- Do not mention a passage, excerpt, chunk, or sentence.
- Do not put an amount, percentage, or share count in the question. Years are allowed.
- expected_answer is a short answer, not a copy of the passage.
- Every figure in expected_answer must be printed in the passage.
- claims is zero to two short lowercase phrases copied from the passage, skipping pure numbers.
- Return every input id once.
"""


@dataclass(frozen=True)
class Fact:
    chunk_id: str
    ticker: str
    company: str
    year: int
    excerpt: str
    key: str

    @property
    def passage(self) -> str:
        return " ".join(self.excerpt.split())


def existing_questions() -> set[str]:
    found: set[str] = set()
    for path in OLD_FILES:
        bank = json.loads(path.read_text(encoding="utf-8"))
        found.update(str(item["question"]).casefold() for item in bank["questions"])
    return found


def load_facts() -> list[Fact]:
    engine = database_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT c.id::text, d.ticker, d.company_name, d.fiscal_year, c.content
                FROM document_chunks c
                JOIN source_documents d ON d.id = c.document_id
                WHERE d.filing_type = '10-K'
                """
            )
        ).all()
    seen: set[str] = set()
    facts: list[Fact] = []
    for chunk_id, ticker, company, year, content in rows:
        if content.count("|") > 4:
            continue
        for match in SENTENCE.finditer(content):
            excerpt = match.group(0).strip()
            passage = " ".join(excerpt.split())
            if SKIP.search(passage):
                continue
            if not FIGURES.search(passage):
                continue
            if sum(ch.isalpha() for ch in passage) < 40:
                continue
            key = passage.casefold()
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                Fact(str(chunk_id), str(ticker), str(company), int(year), excerpt, key)
            )
    return facts


def topic_of(passage: str) -> str:
    lowered = passage.casefold()
    for topic in TOPICS:
        if topic in lowered:
            return topic
    return ""


def has_amount(passage: str) -> bool:
    return bool(VALUES.findall(passage))


def take_spread(facts: list[Fact], banned: set[str]) -> list[Fact]:
    buckets: dict[tuple[str, int], list[Fact]] = defaultdict(list)
    for fact in facts:
        if fact.key not in banned and has_amount(fact.passage):
            buckets[(fact.ticker, fact.year)].append(fact)
    keys = sorted(buckets)
    ordered: list[Fact] = []
    while any(buckets.values()):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
    return ordered


def temporal_pairs(facts: list[Fact], banned: set[str]) -> list[tuple[Fact, Fact]]:
    by_ticker: dict[str, dict[int, list[Fact]]] = defaultdict(lambda: defaultdict(list))
    for fact in take_spread(facts, banned):
        by_ticker[fact.ticker][fact.year].append(fact)
    pairs: list[tuple[Fact, Fact]] = []
    used: set[str] = set()

    def consider(left: Fact, right: Fact, shared_topic: bool) -> None:
        if left.key in used or right.key in used or left.key == right.key:
            return
        if left.chunk_id == right.chunk_id or left.year == right.year:
            return
        earlier, later = (left, right) if left.year < right.year else (right, left)
        if not _distinct_amounts(earlier.passage, later.passage):
            return
        if shared_topic and topic_of(earlier.passage) != topic_of(later.passage):
            return
        if shared_topic and not topic_of(earlier.passage):
            return
        if len(pairs) >= 320:
            return
        used.add(earlier.key)
        used.add(later.key)
        pairs.append((earlier, later))

    for years in by_ticker.values():
        ordered_years = sorted(years)
        for index, year in enumerate(ordered_years):
            for later_year in ordered_years[index + 1 :]:
                for earlier in years[year]:
                    for later in years[later_year]:
                        consider(earlier, later, shared_topic=True)
                        if earlier.key in used:
                            break
    if len(pairs) < TARGETS["temporal"] + 40:
        for years in by_ticker.values():
            ordered_years = sorted(years)
            for index, year in enumerate(ordered_years):
                for later_year in ordered_years[index + 1 :]:
                    for earlier in years[year]:
                        for later in years[later_year]:
                            consider(earlier, later, shared_topic=False)
                            if earlier.key in used:
                                break
    return pairs


def _distinct_amounts(earlier: str, later: str) -> bool:
    left = set(VALUES.findall(earlier))
    right = set(VALUES.findall(later))
    return bool(left - right) and bool(right - left)


def multihop_pairs(facts: list[Fact], banned: set[str]) -> list[tuple[Fact, Fact]]:
    free = [fact for fact in take_spread(facts, banned)]
    used: set[str] = set()
    pairs: list[tuple[Fact, Fact]] = []
    grouped: dict[tuple[str, int], list[Fact]] = defaultdict(list)
    for fact in free:
        grouped[(fact.ticker, fact.year)].append(fact)
    for group in grouped.values():
        if len(pairs) >= 140:
            break
        pending = [fact for fact in group if fact.key not in used]
        for left, right in zip(pending[::2], pending[1::2], strict=False):
            if left.chunk_id == right.chunk_id:
                continue
            used.add(left.key)
            used.add(right.key)
            pairs.append((left, right))
            if len(pairs) >= 140:
                break
    by_ticker: dict[str, list[Fact]] = defaultdict(list)
    for fact in free:
        if fact.key not in used:
            by_ticker[fact.ticker].append(fact)
    tickers = sorted(by_ticker)
    while sum(len(items) for items in by_ticker.values()) >= 2 and len(pairs) < 280:
        progressed = False
        for index, ticker in enumerate(tickers):
            other = tickers[(index + 1) % len(tickers)]
            if ticker == other or not by_ticker[ticker] or not by_ticker[other]:
                continue
            left = by_ticker[ticker].pop(0)
            right = by_ticker[other].pop(0)
            if left.key == right.key or left.chunk_id == right.chunk_id:
                continue
            used.add(left.key)
            used.add(right.key)
            pairs.append((left, right))
            progressed = True
            if len(pairs) >= 280:
                return pairs
        if not progressed:
            break
    return pairs


def client() -> OpenAI:
    return OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=NVIDIA_BASE_URL,
        timeout=90.0,
        max_retries=0,
    )


def chat_items(prompt: str) -> list[dict[str, object]]:
    last_error = "no response"
    for attempt in range(1, 5):
        try:
            response = client().chat.completions.create(
                model=settings.nvidia_chat_model,
                temperature=0.2,
                max_tokens=2200,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            parsed = json.loads(raw)
            items = parsed.get("items", parsed if isinstance(parsed, list) else None)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
            last_error = "items missing"
        except (APITimeoutError, APIConnectionError, json.JSONDecodeError) as exc:
            last_error = type(exc).__name__
        except APIStatusError as exc:
            last_error = f"HTTP {exc.status_code}"
            if exc.status_code not in {408, 429, 500, 502, 503, 504}:
                raise
        time.sleep(min(8 * attempt, 30))
    print(f"  batch failed: {last_error}", flush=True)
    return []


def single_prompt(category: str, facts: list[Fact]) -> str:
    items = [
        {
            "id": str(index),
            "category": category,
            "company": fact.company,
            "fiscal_year": fact.year,
            "passage": fact.passage,
        }
        for index, fact in enumerate(facts)
    ]
    extra = {
        "factual": "Ask who, what, where, or which. Include one claim phrase in the answer.",
        "numerical": "Ask for the amount, count, rate, or percentage. The answer must contain that figure.",
    }[category]
    return json.dumps({"instructions": extra, "items": items}, ensure_ascii=False)


def pair_prompt(category: str, pairs: list[tuple[Fact, Fact]]) -> str:
    items = []
    for index, (left, right) in enumerate(pairs):
        if category == "temporal":
            items.append(
                {
                    "id": str(index),
                    "category": category,
                    "company": left.company,
                    "earlier_fiscal_year": left.year,
                    "later_fiscal_year": right.year,
                    "earlier_passage": left.passage,
                    "later_passage": right.passage,
                }
            )
        else:
            items.append(
                {
                    "id": str(index),
                    "category": category,
                    "passages": [
                        {
                            "company": left.company,
                            "fiscal_year": left.year,
                            "passage": left.passage,
                        },
                        {
                            "company": right.company,
                            "fiscal_year": right.year,
                            "passage": right.passage,
                        },
                    ],
                }
            )
    extra = {
        "temporal": (
            "Ask one question that needs both years, in chronological order. "
            "The answer must state the earlier figure, then the later figure."
        ),
        "multi_hop": "Ask one question that needs both passages. Use a figure from each passage in the answer.",
    }[category]
    return json.dumps({"instructions": extra, "items": items}, ensure_ascii=False)


def figures_supported(answer: str, passages: list[str], years: set[int]) -> bool:
    sources = passages + [str(year) for year in years]
    return all(
        any(figure in source for source in sources) for figure in FIGURES.findall(answer)
    )


def clean_claims(raw: object, passages: list[str], answer: str) -> list[str]:
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = [str(item) for item in raw]
    else:
        values = []
    kept: list[str] = []
    lowered_sources = [passage.casefold() for passage in passages]
    lowered_answer = answer.casefold()
    for value in values:
        claim = " ".join(value.split()).casefold()
        if not 6 <= len(claim) <= 80 or claim in WEAK_CLAIMS or claim in kept:
            continue
        if claim not in lowered_answer:
            continue
        if any(claim in source for source in lowered_sources):
            kept.append(claim)
        if len(kept) == 2:
            break
    return kept


def leaks_amount(question: str, answer: str) -> bool:
    return any(figure in question for figure in VALUES.findall(answer))


def scopes_match(question: str, expected: set[tuple[str, int]]) -> bool:
    found = {
        (scope.ticker, scope.fiscal_year)
        for scope in question_scopes(question)
        if scope.ticker is not None and scope.fiscal_year is not None
    }
    return found == expected


def accept_single(
    category: str,
    fact: Fact,
    item: dict[str, object],
    seen_questions: set[str],
) -> dict[str, object] | None:
    question = " ".join(str(item.get("question", "")).split())
    answer = " ".join(str(item.get("expected_answer", "")).split())
    if not _question_ok(question, seen_questions):
        return None
    if not answer or len(answer) > 320 or answer == REFUSAL:
        return None
    passages = [fact.passage]
    years = {fact.year}
    if not figures_supported(answer, passages, years):
        return None
    if leaks_amount(question, answer):
        return None
    if not scopes_match(question, {(fact.ticker, fact.year)}):
        return None
    claims = clean_claims(item.get("claims"), passages, answer)
    if category == "factual" and not claims and not VALUES.findall(answer):
        return None
    if category == "numerical" and not VALUES.findall(answer):
        return None
    return _record(category, question, answer, (fact,), claims)


def accept_pair(
    category: str,
    pair: tuple[Fact, Fact],
    item: dict[str, object],
    seen_questions: set[str],
) -> dict[str, object] | None:
    left, right = pair
    question = " ".join(str(item.get("question", "")).split())
    answer = " ".join(str(item.get("expected_answer", "")).split())
    if not _question_ok(question, seen_questions):
        return None
    if not answer or len(answer) > 420 or answer == REFUSAL:
        return None
    passages = [left.passage, right.passage]
    years = {left.year, right.year}
    if not figures_supported(answer, passages, years):
        return None
    if leaks_amount(question, answer):
        return None
    expected = {(left.ticker, left.year), (right.ticker, right.year)}
    if not scopes_match(question, expected):
        return None
    amounts = VALUES.findall(answer)
    if category == "temporal":
        if len(amounts) < 2:
            return None
        if amounts[0] not in left.passage or amounts[-1] not in right.passage:
            return None
    else:
        left_amounts = set(VALUES.findall(left.passage))
        right_amounts = set(VALUES.findall(right.passage))
        if not (set(amounts) & left_amounts) or not (set(amounts) & right_amounts):
            return None
    claims = clean_claims(item.get("claims"), passages, answer)
    return _record(category, question, answer, pair, claims)


def _question_ok(question: str, seen_questions: set[str]) -> bool:
    words = question.split()
    if not 8 <= len(words) <= 55:
        return False
    folded = question.casefold()
    if folded in seen_questions:
        return False
    if any(token in folded for token in ("passage", "excerpt", "chunk", "sentence below")):
        return False
    return "fiscal" in folded


def _record(
    category: str,
    question: str,
    answer: str,
    facts: tuple[Fact, ...],
    claims: list[str],
) -> dict[str, object]:
    evidence = [
        {
            "chunk_id": fact.chunk_id,
            "ticker": fact.ticker,
            "company_name": fact.company,
            "fiscal_year": fact.year,
            "filing_type": "10-K",
            "excerpt": fact.excerpt,
        }
        for fact in facts
    ]
    record: dict[str, object] = {
        "id": "",
        "category": category,
        "question": question,
        "expected_answer": answer,
        "chunk_ids": list(dict.fromkeys(fact.chunk_id for fact in facts)),
        "evidence": evidence,
    }
    if claims:
        record["claims"] = claims
    return record


def absent(conn, name: str) -> bool:
    in_docs = conn.execute(
        text(
            """
            SELECT count(*) FROM source_documents
            WHERE company_name ILIKE :name OR ticker ILIKE :name
            """
        ),
        {"name": f"%{name}%"},
    ).scalar_one()
    if int(in_docs) != 0:
        return False
    pattern = r"\m" + re.escape(name) + r"\M" if " " not in name else re.escape(name)
    in_chunks = conn.execute(
        text(
            """
            SELECT count(*) FROM document_chunks
            WHERE content ~* :pattern
            """
        ),
        {"pattern": pattern},
    ).scalar_one()
    return int(in_chunks) == 0


def unanswerable(seen_questions: set[str]) -> list[dict[str, object]]:
    engine = database_engine()
    records: list[dict[str, object]] = []
    years = (2022, 2023, 2024, 2025)
    with engine.connect() as conn:
        usable = [name for name in OUTSIDERS if absent(conn, name)]
    print(f"absent companies: {len(usable)}", flush=True)
    cursor = 0
    for name in usable:
        for template in METRIC_TEMPLATES:
            year = years[cursor % len(years)]
            cursor += 1
            question = template.format(name=name, year=year)
            if question.casefold() in seen_questions:
                continue
            if any(alias in question.casefold() for alias in ("apple", "amazon", "microsoft", "nvidia", "alphabet", "google")):
                continue
            records.append(
                {
                    "id": "",
                    "category": "unanswerable",
                    "question": question,
                    "expected_answer": REFUSAL,
                    "chunk_ids": [],
                    "evidence": [],
                }
            )
            seen_questions.add(question.casefold())
            if len(records) == TARGETS["unanswerable"]:
                return records
    raise SystemExit(f"only {len(records)} unanswerable questions passed the absence checks")


def load_checkpoint() -> list[dict[str, object]]:
    if not CHECKPOINT.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_checkpoint(record: dict[str, object], keys: list[str], lock: threading.Lock) -> None:
    line = json.dumps({"record": record, "keys": keys}, ensure_ascii=False)
    with lock, CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def assign_ids(records: list[dict[str, object]]) -> None:
    counts = dict.fromkeys(TARGETS, 0)
    order = {category: index for index, category in enumerate(TARGETS)}
    records.sort(key=lambda record: (order[str(record["category"])], str(record["question"])))
    for record in records:
        category = str(record["category"])
        counts[category] += 1
        record["id"] = f"{PREFIX[category]}-{counts[category]:03d}"


def write_output(records: list[dict[str, object]]) -> None:
    assign_ids(records)
    counts: dict[str, int] = {}
    for record in records:
        category = str(record["category"])
        counts[category] = counts.get(category, 0) + 1
    if sum(counts.values()) != 1000 or counts != TARGETS:
        raise SystemExit(f"unexpected counts: {counts}")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate question id")
    questions = [str(record["question"]).casefold() for record in records]
    if len(questions) != len(set(questions)):
        raise SystemExit("duplicate question")
    payload = {
        "description": (
            "One thousand questions. None of them repeat ground_truth.json or queries_100.json. "
            "Answerable excerpts are exact slices of document_chunks. "
            "A correct answer must contain every figure in expected_answer and every claim."
        ),
        "refusal": REFUSAL,
        "counts": counts,
        "questions": records,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name} {counts}", flush=True)


def _by_id(items: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for item in items:
        found[str(item.get("id", ""))] = item
    return found


def generate_singles(
    category: str,
    queue: list[Fact],
    seen_questions: set[str],
    done: list[dict[str, object]],
    lock: threading.Lock,
) -> None:
    target = TARGETS[category]
    cursor = 0

    def work(batch: list[Fact]) -> list[dict[str, object]]:
        return chat_items(single_prompt(category, batch))

    _drain(
        category,
        queue,
        cursor,
        target,
        done,
        seen_questions,
        lock,
        work,
        lambda batch, items: _zip_singles(category, batch, items, seen_questions),
    )


def _zip_singles(
    category: str,
    batch: list[Fact],
    items: list[dict[str, object]],
    seen_questions: set[str],
) -> list[tuple[dict[str, object], list[str]]]:
    found = _by_id(items)
    accepted: list[tuple[dict[str, object], list[str]]] = []
    for index, fact in enumerate(batch):
        item = found.get(str(index))
        if item is None:
            continue
        record = accept_single(category, fact, item, seen_questions)
        if record is None:
            continue
        accepted.append((record, [fact.key]))
    return accepted


def generate_pairs(
    category: str,
    queue: list[tuple[Fact, Fact]],
    seen_questions: set[str],
    done: list[dict[str, object]],
    lock: threading.Lock,
) -> None:
    def work(batch: list[tuple[Fact, Fact]]) -> list[dict[str, object]]:
        return chat_items(pair_prompt(category, batch))

    _drain(
        category,
        queue,
        0,
        TARGETS[category],
        done,
        seen_questions,
        lock,
        work,
        lambda batch, items: _zip_pairs(category, batch, items, seen_questions),
    )


def _zip_pairs(
    category: str,
    batch: list[tuple[Fact, Fact]],
    items: list[dict[str, object]],
    seen_questions: set[str],
) -> list[tuple[dict[str, object], list[str]]]:
    found = _by_id(items)
    accepted: list[tuple[dict[str, object], list[str]]] = []
    for index, pair in enumerate(batch):
        item = found.get(str(index))
        if item is None:
            continue
        record = accept_pair(category, pair, item, seen_questions)
        if record is None:
            continue
        accepted.append((record, [pair[0].key, pair[1].key]))
    return accepted


def _drain(
    category: str,
    queue: list,
    cursor: int,
    target: int,
    done: list[dict[str, object]],
    seen_questions: set[str],
    lock: threading.Lock,
    work,
    accept,
) -> None:
    inflight: dict[Future, list] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while True:
            have = sum(1 for record in done if record["category"] == category)
            while (
                have < target
                and len(inflight) < WORKERS
                and cursor < len(queue)
                and have + len(inflight) * BATCH < target + BATCH
            ):
                batch = queue[cursor : cursor + BATCH]
                cursor += len(batch)
                inflight[pool.submit(work, batch)] = batch
            if not inflight:
                break
            future = next(as_completed(list(inflight)))
            batch = inflight.pop(future)
            try:
                items = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  {category} worker error: {exc}", flush=True)
                continue
            fresh = 0
            for record, keys in accept(batch, items):
                have_now = sum(1 for row in done if row["category"] == category)
                if have_now >= target:
                    break
                question = str(record["question"]).casefold()
                if question in seen_questions:
                    continue
                seen_questions.add(question)
                done.append(record)
                append_checkpoint(record, keys, lock)
                fresh += 1
            have = sum(1 for record in done if record["category"] == category)
            print(
                f"{category} {have}/{target} (+{fresh}, queue {cursor}/{len(queue)})",
                flush=True,
            )
            if have >= target:
                for pending in inflight:
                    pending.cancel()
                break
    have = sum(1 for record in done if record["category"] == category)
    if have < target:
        raise SystemExit(f"{category} stopped at {have}/{target}")


def build() -> None:
    prior = existing_questions()
    saved = load_checkpoint()
    done: list[dict[str, object]] = []
    banned: set[str] = set()
    seen = set(prior)
    for row in saved:
        record = row["record"]
        question = str(record["question"]).casefold()
        if question in seen:
            continue
        seen.add(question)
        done.append(record)
        banned.update(str(key) for key in row["keys"])
    if done:
        print(f"resumed {len(done)} questions", flush=True)
    counts = {category: sum(1 for record in done if record["category"] == category) for category in TARGETS}
    if counts["unanswerable"] < TARGETS["unanswerable"]:
        rows = unanswerable(seen)
        lock = threading.Lock()
        for record in rows:
            if counts["unanswerable"] >= TARGETS["unanswerable"]:
                break
            done.append(record)
            append_checkpoint(record, [], lock)
            counts["unanswerable"] += 1
        print(f"unanswerable {counts['unanswerable']}/{TARGETS['unanswerable']}", flush=True)
    need_corpus = any(counts[category] < TARGETS[category] for category in ("factual", "numerical", "multi_hop", "temporal"))
    if need_corpus:
        facts = load_facts()
        print(f"narrative sentences: {len(facts)}", flush=True)
        pairs_t = temporal_pairs(facts, banned)
        used_t = {key for pair in pairs_t for key in (pair[0].key, pair[1].key)}
        pairs_h = multihop_pairs(facts, banned | used_t)
        used_h = {key for pair in pairs_h for key in (pair[0].key, pair[1].key)}
        singles = [
            fact
            for fact in take_spread(facts, banned | used_t | used_h)
        ]
        split = max(len(singles) // 2, 1)
        factual_queue = singles[:split]
        numerical_queue = singles[split:]
        print(
            f"queues temporal={len(pairs_t)} multi_hop={len(pairs_h)} "
            f"factual={len(factual_queue)} numerical={len(numerical_queue)}",
            flush=True,
        )
        lock = threading.Lock()
        if counts["temporal"] < TARGETS["temporal"]:
            generate_pairs("temporal", pairs_t, seen, done, lock)
        if counts["multi_hop"] < TARGETS["multi_hop"]:
            generate_pairs("multi_hop", pairs_h, seen, done, lock)
        if counts["numerical"] < TARGETS["numerical"]:
            generate_singles("numerical", numerical_queue, seen, done, lock)
        if counts["factual"] < TARGETS["factual"]:
            generate_singles("factual", factual_queue, seen, done, lock)
    write_output(done)
    CHECKPOINT.unlink(missing_ok=True)


def pilot() -> None:
    """Generate one small batch per answerable category and print what was kept."""
    facts = load_facts()
    seen = existing_questions()
    pairs_t = temporal_pairs(facts, set())
    used_t = {key for pair in pairs_t for key in (pair[0].key, pair[1].key)}
    pairs_h = multihop_pairs(facts, used_t)
    print(f"sentences={len(facts)} temporal={len(pairs_t)} multi_hop={len(pairs_h)}", flush=True)
    for category, batch in (
        ("temporal", pairs_t[:BATCH]),
        ("multi_hop", pairs_h[:BATCH]),
    ):
        items = chat_items(pair_prompt(category, batch))
        kept = _zip_pairs(category, batch, items, seen)
        print(f"{category} kept {len(kept)}/{len(batch)}", flush=True)
        if kept:
            print(kept[0][0]["question"], flush=True)
            print(kept[0][0]["expected_answer"], flush=True)
        elif items:
            print("sample", items[0], flush=True)
    used_h = {key for pair in pairs_h for key in (pair[0].key, pair[1].key)}
    singles = take_spread(facts, used_t | used_h)
    for category, batch in (
        ("numerical", singles[:BATCH]),
        ("factual", singles[BATCH : BATCH * 2]),
    ):
        items = chat_items(single_prompt(category, batch))
        kept = _zip_singles(category, batch, items, seen)
        print(f"{category} kept {len(kept)}/{len(batch)}", flush=True)
        if kept:
            print(kept[0][0]["question"], flush=True)
            print(kept[0][0]["expected_answer"], flush=True)
        elif items:
            print("sample", items[0], flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        pilot()
    else:
        build()
