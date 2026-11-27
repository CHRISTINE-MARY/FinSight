from pydantic_ai.exceptions import ModelQuotaExceededError, ModelRateLimitError

from app.chat.orchestrator import AssistantError
from app.grounding.validator import GroundingError
from evals.run_eval import (
    _grounding_outcome,
    _is_quota,
    _is_rate_limit,
    _wait_seconds,
)


def _wrap(cause: BaseException) -> AssistantError:
    error = AssistantError("The assistant could not complete this answer")
    error.__cause__ = cause
    return error


def test_rate_limit_waits_for_the_provider_delay() -> None:
    error = _wrap(ModelRateLimitError("model", "slow down", retry_after=12))
    assert _is_rate_limit(error)
    assert _wait_seconds(error, 1) == 12


def test_rate_limit_backs_off_when_the_provider_gives_no_delay() -> None:
    error = _wrap(ModelRateLimitError("model", "slow down"))
    assert _wait_seconds(error, 1) == 8
    assert _wait_seconds(error, 2) == 16
    assert _wait_seconds(error, 3) == 32


def test_http_429_is_a_rate_limit() -> None:
    class Rejected(Exception):
        status_code = 429

    assert _is_rate_limit(_wrap(Rejected("too many requests")))


def test_quota_is_not_retried_as_a_rate_limit() -> None:
    error = _wrap(ModelQuotaExceededError("model", "no credits"))
    assert _is_quota(error)
    assert not _is_rate_limit(error)


def test_other_failures_wait_two_seconds() -> None:
    assert _wait_seconds(TimeoutError(), 1) == 2


def test_a_wrapped_claim_keeps_its_grounding_kind() -> None:
    citation = GroundingError(
        "Claim 'One Microsoft Way.' failed: Citation mistake: "
        "Locator is not in the retrieved passage"
    )
    invented = GroundingError(
        "Claim '228,000 employees.' failed: Hallucination: "
        "Locator is not in the retrieved passage"
    )
    other = GroundingError(
        "Claim '164,000 employees.' failed: "
        "The cited passage does not support the answer"
    )

    assert _grounding_outcome(citation) == "citation_mistake"
    assert _grounding_outcome(invented) == "hallucination"
    assert _grounding_outcome(other) == "grounding_rejected"
