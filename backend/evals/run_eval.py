"""Run the 50 ground-truth questions and write evals/results.json.

From backend/:

    python evals/run_eval.py
    python evals/run_eval.py results.json queries_100.json

Each question is scored once. A turn has a 180s timeout. Transport
failures and rate limits are retried; a grounding rejection is recorded.
Does not save chats.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.auth.dependencies import CurrentUser
from app.chat.orchestrator import AssistantError, RetrievalError, run_turn
from app.database.chats import ThreadRecord
from app.grounding.validator import GroundingError, says_not_enough
from app.retrieval.retriever import DocumentRetriever

ROOT = Path(__file__).resolve().parent
GROUND_TRUTH = ROOT / "ground_truth.json"
RESULTS = ROOT / "results.json"
AT_K = 8
ATTEMPTS = 3
QUESTION_TIMEOUT_S = 180
TRANSIENT_SLEEP_S = 2.0
RATE_LIMIT_BACKOFF_S = (8.0, 16.0, 32.0)
RATE_LIMIT_WAIT_CAP_S = 60.0

FIGURES = __import__("re").compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+\.\d+%?|\d+%|\b(?:19|20)\d{2}\b"
)
VALUES = __import__("re").compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+\.\d+%?|\d+%"
)
# Qualitative claims the figure check cannot see.
CLAIMS: dict[str, tuple[str, ...]] = {
    "fact-01": ("saturday", "september"),
    "fact-02": ("google services", "google cloud", "other bets"),
    "fact-03": ("north america", "international", "aws"),
    "fact-06": ("flat",),
    "fact-07": ("mac",),
}
FACT_07_DIRECTION = ("increas", "rose", "higher", "grew", "growth")


async def _empty_history(user: CurrentUser, thread_id: object) -> list[object]:
    return []


def answer_correct(item: dict[str, object], text: str) -> bool:
    if not text or says_not_enough(text):
        return False
    expected = str(item["expected_answer"])
    if any(figure not in text for figure in FIGURES.findall(expected)):
        return False
    if item["category"] == "temporal":
        cursor = 0
        for value in VALUES.findall(expected):
            found = text.find(value, cursor)
            if found < 0:
                return False
            cursor = found + len(value)
    lowered = text.lower()
    extra = item.get("claims") or ()
    phrases = list(CLAIMS.get(str(item["id"]), ())) + list(extra)
    for phrase in phrases:
        if phrase == "aws":
            if "aws" not in lowered and "amazon web services" not in lowered:
                return False
            continue
        if phrase not in lowered:
            return False
    if item["id"] == "fact-07":
        return any(word in lowered for word in FACT_07_DIRECTION)
    return True


def retrieval_scores(retrieved: list[str], gold: list[str]) -> tuple[float, float, float]:
    gold_set = set(gold)
    top = retrieved[:AT_K]
    hit = len(gold_set & set(top))
    recall = hit / len(gold_set)
    precision = hit / AT_K
    rank = next((index for index, chunk_id in enumerate(top, start=1) if chunk_id in gold_set), None)
    reciprocal = 0.0 if rank is None else 1.0 / rank
    return recall, reciprocal, precision


def citation_precision(cited: list[str], gold: list[str]) -> float:
    unique = list(dict.fromkeys(cited))
    if not unique:
        return 0.0
    return len(set(gold) & set(unique)) / len(unique)


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(percent / 100 * len(ordered))
    return ordered[rank - 1]


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_rate_limit(exc: BaseException) -> bool:
    for item in _chain(exc):
        if getattr(item, "status_code", None) == 429:
            return True
        if "RateLimit" in type(item).__name__:
            return True
    return False


def _is_quota(exc: BaseException) -> bool:
    return any(type(item).__name__ == "ModelQuotaExceededError" for item in _chain(exc))


def _retry_after(exc: BaseException) -> float | None:
    for item in _chain(exc):
        value = getattr(item, "retry_after", None)
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
            return min(float(value), RATE_LIMIT_WAIT_CAP_S)
        response = getattr(item, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            continue
        raw = headers.get("retry-after")
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            return min(seconds, RATE_LIMIT_WAIT_CAP_S)
    return None


def _wait_seconds(exc: BaseException, attempt: int) -> float:
    if _is_rate_limit(exc):
        announced = _retry_after(exc)
        if announced is not None:
            return announced
        index = min(attempt - 1, len(RATE_LIMIT_BACKOFF_S) - 1)
        return RATE_LIMIT_BACKOFF_S[index]
    return TRANSIENT_SLEEP_S


def _grounding_outcome(exc: GroundingError) -> str:
    text = str(exc)
    # The turn prefixes the failing claim, so the reason is not always the start.
    if "Citation mistake:" in text:
        return "citation_mistake"
    if "Hallucination:" in text:
        return "hallucination"
    return "grounding_rejected"


def _failure_text(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError) and not str(exc):
        text = f"TimeoutError: exceeded {QUESTION_TIMEOUT_S}s"
    else:
        text = f"{type(exc).__name__}: {exc}"
    cause = exc.__cause__
    if cause is not None:
        text = f"{text} ({type(cause).__name__}: {cause})"
    return text


async def ask(
    question: str, search: SearchCapture, label: str
) -> tuple[str, list[str], str, float, list[str]]:
    from app.chat import orchestrator

    orchestrator.list_messages = _empty_history
    user = CurrentUser(id=uuid4(), email="eval@localhost", access_token="eval")
    now = datetime.now(UTC)
    thread = ThreadRecord(id=uuid4(), title="eval", created_at=now, updated_at=now)
    last_error = "failed"
    last_elapsed = 0.0
    last_ids: list[str] = []
    for attempt in range(1, ATTEMPTS + 1):
        search.reset()
        started = time.perf_counter()
        try:
            turn = await asyncio.wait_for(
                run_turn(user, thread, question),
                timeout=QUESTION_TIMEOUT_S,
            )
        except GroundingError as exc:
            return (
                str(exc),
                [],
                _grounding_outcome(exc),
                time.perf_counter() - started,
                search.ids,
            )
        except (RetrievalError, AssistantError, TimeoutError) as exc:
            last_error = _failure_text(exc)
            last_elapsed = time.perf_counter() - started
            last_ids = list(search.ids)
            if _is_quota(exc) or attempt == ATTEMPTS:
                return "", [], last_error, last_elapsed, last_ids
            wait = _wait_seconds(exc, attempt)
            reason = "rate limit" if _is_rate_limit(exc) else "transient error"
            print(
                f"  {label} retry {attempt}/{ATTEMPTS} in {wait:.0f}s ({reason})",
                flush=True,
            )
            await asyncio.sleep(wait)
            continue
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            traceback.print_exc()
            return "", [], _failure_text(exc), elapsed, search.ids
        cited = [str(citation.chunk_id) for citation in turn.citations]
        return turn.answer, cited, "answered", time.perf_counter() - started, search.ids
    return "", [], last_error, last_elapsed, last_ids


class SearchCapture:
    """Remember the first top-8 list. Later tool searches stay out of Recall@8."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self._original = DocumentRetriever.search

    def reset(self) -> None:
        self.ids = []

    def install(self) -> None:
        original = self._original

        def wrapped(retriever: DocumentRetriever, question: str, **kwargs: object):
            passages = original(retriever, question, **kwargs)
            if not self.ids:
                self.ids = [str(passage.chunk_id) for passage in passages[:AT_K]]
            return passages

        DocumentRetriever.search = wrapped  # type: ignore[method-assign]

    def remove(self) -> None:
        DocumentRetriever.search = self._original  # type: ignore[method-assign]


def score_row(
    item: dict[str, object],
    text: str,
    cited: list[str],
    retrieved: list[str],
    outcome: str,
    elapsed: float,
) -> dict[str, object]:
    gold = [str(chunk_id) for chunk_id in item["chunk_ids"]]  # type: ignore[index]
    refused = bool(text) and says_not_enough(text) and not cited
    answerable = item["category"] != "unanswerable"
    if answerable:
        recall, reciprocal, precision_at = retrieval_scores(retrieved, gold)
        cite = citation_precision(cited, gold)
        accurate = outcome == "answered" and answer_correct(item, text)
        abstained = 0.0 if refused else 1.0
    else:
        recall = reciprocal = precision_at = cite = None
        accurate = refused
        abstained = 1.0 if refused else 0.0
    return {
        "id": item["id"],
        "category": item["category"],
        "outcome": outcome,
        "latency_s": round(elapsed, 3),
        "recall_at_8": round_metric(recall) if recall is not None else None,
        "reciprocal_rank_at_8": round_metric(reciprocal) if reciprocal is not None else None,
        "precision_at_8": round_metric(precision_at) if precision_at is not None else None,
        "answer_accuracy": 1.0 if accurate else 0.0,
        "citation_precision": round_metric(cite) if cite is not None else None,
        "abstention_accuracy": abstained,
        "gold_chunks": item.get("evidence", []),
        "retrieved": retrieved,
        "cited": list(dict.fromkeys(cited)),
        "answer": text,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    def pack(group: list[dict[str, object]]) -> dict[str, object]:
        answerable = [row for row in group if row["category"] != "unanswerable"]
        outcomes = [str(row["outcome"]) for row in group]
        return {
            "n": len(group),
            "citation_mistake": outcomes.count("citation_mistake"),
            "hallucination": outcomes.count("hallucination"),
            "Recall@8": round_metric(mean([float(row["recall_at_8"]) for row in answerable])),
            "MRR@8": round_metric(
                mean([float(row["reciprocal_rank_at_8"]) for row in answerable])
            ),
            "Precision@8": round_metric(
                mean([float(row["precision_at_8"]) for row in answerable])
            ),
            "Answer Accuracy": round_metric(
                mean([float(row["answer_accuracy"]) for row in group])
            ),
            "Citation Precision": round_metric(
                mean(
                    [
                        float(row["citation_precision"])
                        for row in answerable
                        if row["citation_precision"] is not None
                    ]
                )
            ),
            "Abstention Accuracy": round_metric(
                mean([float(row["abstention_accuracy"]) for row in group])
            ),
            "P95 Latency": round_metric(
                percentile([float(row["latency_s"]) for row in group], 95)
            ),
        }

    by_category = {
        category: pack([row for row in rows if row["category"] == category])
        for category in ("factual", "numerical", "multi_hop", "temporal", "unanswerable")
    }
    return {"overall": pack(rows), "by_category": by_category}


async def main() -> None:
    out = ROOT / (sys.argv[1] if len(sys.argv) > 1 else RESULTS.name)
    bank_path = ROOT / (sys.argv[2] if len(sys.argv) > 2 else GROUND_TRUTH.name)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    questions = bank["questions"]
    if bank_path == GROUND_TRUTH and len(questions) != 50:
        raise SystemExit(f"expected 50 questions, found {len(questions)}")
    if not questions:
        raise SystemExit(f"no questions in {bank_path.name}")
    print(f"loaded {len(questions)} questions from {bank_path.name}", flush=True)
    capture = SearchCapture()
    capture.install()
    rows: list[dict[str, object]] = []
    try:
        total = len(questions)
        for index, item in enumerate(questions, start=1):
            label = str(item["id"])
            text, cited, outcome, elapsed, retrieved = await ask(
                str(item["question"]), capture, label
            )
            row = score_row(item, text, cited, retrieved, outcome, elapsed)
            rows.append(row)
            out.write_text(
                json.dumps(
                    {"completed": index, "summary": summarize(rows), "questions": rows},
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"{index}/{total} {row['id']} {outcome} "
                f"recall={row['recall_at_8']} acc={row['answer_accuracy']} "
                f"{row['latency_s']}s",
                flush=True,
            )
    finally:
        capture.remove()
    payload = {
        "scoring": {
            "Recall@8": (
                "share of gold chunks in the first search's top 8, "
                "averaged over answerable questions"
            ),
            "MRR@8": "1/rank of the first gold chunk in that top 8, else 0",
            "Precision@8": "gold chunks in the top 8, divided by 8",
            "Answer Accuracy": (
                "every gold figure is present; temporal values stay in gold "
                "order; unanswerable items must refuse"
            ),
            "Citation Precision": (
                "share of cited chunks that are gold, averaged over "
                "answerable questions"
            ),
            "Abstention Accuracy": (
                "answerable questions that did not refuse, plus unanswerable "
                "questions that refused with no citation"
            ),
            "P95 Latency": "nearest-rank 95th percentile of full-turn seconds",
        },
        "summary": summarize(rows),
        "questions": rows,
    }
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("EVAL DONE", flush=True)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
