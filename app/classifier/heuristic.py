"""v1 complexity classifier: rule-based, no training data required.

This exists to get end-to-end routing working immediately. It is
deliberately simple and will misjudge borderline prompts sometimes — that's
expected and documented, not a bug to chase down. Once
data/labeled_prompts.csv has 200+ hand-labeled rows, a v2 scikit-learn
classifier replaces the scoring in `classify()` below; it will keep reusing
`extract_features()` and the ComplexityTier/ClassificationResult contract,
so nothing else in the app needs to change when that swap happens.
"""

from dataclasses import dataclass, field

from .features import PromptFeatures, extract_features
from .tiers import ComplexityTier

# Score thresholds separating the three tiers. Tuned by hand against a
# handful of example prompts, not a statistical fit — expect to revisit
# once real labeled data exists.
TIER_2_MIN_SCORE = 1
TIER_3_MIN_SCORE = 3

MAX_COMPLEX_KEYWORD_POINTS = 3


@dataclass
class ClassificationResult:
    tier: ComplexityTier
    score: int
    features: PromptFeatures
    matched_signals: list[str] = field(default_factory=list)


def classify(prompt: str) -> ClassificationResult:
    features = extract_features(prompt)
    score = 0
    signals: list[str] = []

    if features.simple_keyword_matches > 0:
        score -= 1
        signals.append(f"{features.simple_keyword_matches} simple-task keyword(s) matched")

    if features.moderate_keyword_matches > 0:
        score += 1
        signals.append(f"{features.moderate_keyword_matches} moderate-task keyword(s) matched")

    if features.complex_keyword_matches > 0:
        points = min(features.complex_keyword_matches, MAX_COMPLEX_KEYWORD_POINTS)
        score += points
        signals.append(f"{features.complex_keyword_matches} complex-task keyword(s) matched (+{points})")

    if features.constraint_count >= 1:
        score += 1
        signals.append(f"constraint_count={features.constraint_count} (>=1)")
    if features.constraint_count >= 3:
        score += 1
        signals.append(f"constraint_count={features.constraint_count} (>=3)")

    if features.word_count > 40:
        score += 1
        signals.append(f"word_count={features.word_count} (>40)")
    if features.word_count > 100:
        score += 1
        signals.append(f"word_count={features.word_count} (>100)")

    if features.has_long_context_block and features.word_count > 30:
        score += 1
        signals.append("long quoted/context block provided alongside a non-trivial prompt")

    if score >= TIER_3_MIN_SCORE:
        tier = ComplexityTier.TIER_3
    elif score >= TIER_2_MIN_SCORE:
        tier = ComplexityTier.TIER_2
    else:
        tier = ComplexityTier.TIER_1

    return ClassificationResult(tier=tier, score=score, features=features, matched_signals=signals)
