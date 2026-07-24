"""Phase 8b: routing memory. Retrieve the k most similar PAST logged
requests (from Phase 7's database) to a new prompt, along with their
verified outcomes, as a second signal alongside the Phase 2 classifier.

Same TF-IDF caveat as compression.py - lexical similarity over past
prompt text, not semantic embeddings.

This is advisory only, same as Phase 3's cost prediction and Phase 5's
multi-objective scoring: it produces a signal, it does not automatically
feed back into route_request(). Wiring it in as an actual input to
routing decisions is a further integration step, not done here.
"""

from collections import Counter
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..logging.db import LoggedRequest, get_recent_requests


@dataclass
class SimilarPastRequest:
    logged_request: LoggedRequest
    similarity: float


@dataclass
class RoutingMemorySignal:
    query: str
    history_size: int
    similar: list[SimilarPastRequest]  # sorted most similar first
    escalation_rate_among_similar: float | None  # None if nothing verified among the similar set
    most_common_tier_among_similar: str | None


def get_routing_memory_signal(
    prompt: str, top_k: int = 5, history_limit: int = 500
) -> RoutingMemorySignal:
    history = get_recent_requests(limit=history_limit)

    if not history:
        return RoutingMemorySignal(
            query=prompt, history_size=0, similar=[], escalation_rate_among_similar=None,
            most_common_tier_among_similar=None,
        )

    corpus = [row.prompt_text for row in history]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([prompt] + corpus)
        similarities = cosine_similarity(matrix[0:1], matrix[1:])[0]
    except ValueError:
        # Degenerate corpus (e.g. everything is stopwords) - no meaningful signal.
        return RoutingMemorySignal(
            query=prompt, history_size=len(history), similar=[], escalation_rate_among_similar=None,
            most_common_tier_among_similar=None,
        )

    ranked = sorted(zip(history, similarities), key=lambda pair: pair[1], reverse=True)[:top_k]
    similar = [SimilarPastRequest(logged_request=row, similarity=float(sim)) for row, sim in ranked]

    verified_similar = [s for s in similar if s.logged_request.verification_status is not None]
    escalation_rate = (
        sum(1 for s in verified_similar if s.logged_request.escalated) / len(verified_similar)
        if verified_similar
        else None
    )

    most_common_tier = Counter(s.logged_request.tier for s in similar).most_common(1)[0][0] if similar else None

    return RoutingMemorySignal(
        query=prompt,
        history_size=len(history),
        similar=similar,
        escalation_rate_among_similar=escalation_rate,
        most_common_tier_among_similar=most_common_tier,
    )
