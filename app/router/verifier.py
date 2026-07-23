"""Phase 6: quality verification + auto-escalation.

Compares the routed model's answer against a reference answer from the
strongest configured model (Tier 3's routing target), scores agreement
with a cheap text-similarity metric, and escalates - swaps in the
reference answer - when agreement is too low.

ON "ASYNC": the original design calls for this to run as a fire-and-forget
background job so it never blocks the user-facing response. There's no
job queue or background worker yet (that's Phase 10's FastAPI + Docker +
background worker), so this phase is SYNCHRONOUS: verify_and_escalate()
makes the reference call and returns inline. When Phase 10 adds a real
worker, this function is what gets queued rather than called directly -
the verification logic itself doesn't need to change.

ON SCORING: this uses difflib text similarity, not an LLM-as-judge, even
though the original design names both. Text similarity is a weak proxy
for semantic agreement - two answers can be worded completely differently
but mean the same thing (scores low here even though it shouldn't), or be
superficially similar but factually wrong (scores high even though it
shouldn't). An LLM-as-judge call would be a meaningfully better signal;
it's a documented, open follow-up rather than something built here.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum

from ..classifier.tiers import ComplexityTier
from ..models.dispatcher import send_request
from ..models.registry import ModelConfig, get_model
from .routing_engine import RoutingResult, load_routing_map

# Below this similarity score, the routed model's answer is treated as a
# likely routing failure and gets escalated to the reference answer.
AGREEMENT_THRESHOLD = 0.5


class VerificationStatus(str, Enum):
    VERIFIED = "verified"                  # agreement met the threshold
    ESCALATED = "escalated"                # agreement below threshold, swapped to reference
    ALREADY_TOP_TIER = "already_top_tier"  # routed model IS the reference model
    COMPARISON_FAILED = "comparison_failed"  # original and/or reference call itself failed


@dataclass
class VerificationResult:
    status: VerificationStatus
    agreement_score: float | None  # None when a comparison couldn't be made
    reference_model: str
    reference_text: str
    reference_cost_usd: float
    final_response_text: str
    final_model: str
    cost_delta_usd: float  # extra cost incurred by making the reference call

    @property
    def escalated(self) -> bool:
        return self.status == VerificationStatus.ESCALATED


def _reference_model() -> ModelConfig:
    """The model Tier 3 routes to - used as ground truth to check a
    cheaper model's answer against.
    """
    provider, model_id = load_routing_map()[ComplexityTier.TIER_3]
    return get_model(provider, model_id)


def score_agreement(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def verify_and_escalate(
    routing_result: RoutingResult, threshold: float = AGREEMENT_THRESHOLD
) -> VerificationResult:
    original = routing_result.response
    reference_model = _reference_model()

    if reference_model.key == routing_result.model_config.key:
        return VerificationResult(
            status=VerificationStatus.ALREADY_TOP_TIER,
            agreement_score=None,
            reference_model=reference_model.key,
            reference_text=original.text,
            reference_cost_usd=0.0,
            final_response_text=original.text,
            final_model=routing_result.model_config.key,
            cost_delta_usd=0.0,
        )

    reference_response = send_request(routing_result.prompt, reference_model)

    if not original.ok or not reference_response.ok:
        # Can't meaningfully compare if either call failed outright. Escalate
        # only if the reference succeeded where the original didn't - don't
        # discard a working response just because the verification call failed.
        escalate = (not original.ok) and reference_response.ok
        final_text = reference_response.text if escalate else original.text
        final_model = reference_model.key if escalate else routing_result.model_config.key
        return VerificationResult(
            status=VerificationStatus.ESCALATED if escalate else VerificationStatus.COMPARISON_FAILED,
            agreement_score=None,
            reference_model=reference_model.key,
            reference_text=reference_response.text,
            reference_cost_usd=reference_response.cost_usd,
            final_response_text=final_text,
            final_model=final_model,
            cost_delta_usd=reference_response.cost_usd,
        )

    agreement = score_agreement(original.text, reference_response.text)
    escalate = agreement < threshold
    final_text = reference_response.text if escalate else original.text
    final_model = reference_model.key if escalate else routing_result.model_config.key

    return VerificationResult(
        status=VerificationStatus.ESCALATED if escalate else VerificationStatus.VERIFIED,
        agreement_score=agreement,
        reference_model=reference_model.key,
        reference_text=reference_response.text,
        reference_cost_usd=reference_response.cost_usd,
        final_response_text=final_text,
        final_model=final_model,
        cost_delta_usd=reference_response.cost_usd,
    )
