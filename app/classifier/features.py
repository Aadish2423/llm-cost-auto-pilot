"""Feature extraction shared by the v1 heuristic classifier and the future
v2 scikit-learn classifier (Phase 2b, once data/labeled_prompts.csv has
200+ hand-labeled rows). Only the scoring logic differs between v1 and v2 —
both start from the same PromptFeatures.
"""

import re
from dataclasses import dataclass

SIMPLE_KEYWORDS = [
    "extract", "reformat", "convert", "translate", "rewrite",
    "what is", "capital of",
    # NOTE: bare "list" was deliberately removed — it's a noun far more
    # often than an instruction ("...the expected list"), and was pulling
    # the score down on prompts that just happen to mention a list.
]
MODERATE_KEYWORDS = [
    "summarize", "summarise", "summary", "classify", "classification",
    "categorize", "categorise", "organize", "organise", "structure",
]
COMPLEX_KEYWORDS = [
    "analyze", "analyse", "compare", "evaluate", "synthesize", "synthesise",
    "critique", "design", "argue", "justify", "recommend", "trade-off",
    "tradeoff", "debug", "walk through", "step by step", "creative",
]
CONSTRAINT_PATTERNS = [
    r"\bmust\b", r"\bshould\b", r"\bbudget\b", r"\bunder \$",
    r"\bconstraint", r"\brequire[sd]?\b",
]
CONTEXT_BLOCK_PATTERN = re.compile(r"['\"][^'\"]{15,}['\"]")


def _count_keyword_matches(lowered: str, keywords: list[str]) -> int:
    # Word-boundary matching, not plain substring — otherwise a keyword
    # like "list" would also match inside unrelated words like "checklist".
    return sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", lowered))


@dataclass
class PromptFeatures:
    word_count: int
    simple_keyword_matches: int
    moderate_keyword_matches: int
    complex_keyword_matches: int
    constraint_count: int
    has_long_context_block: bool


def extract_features(prompt: str) -> PromptFeatures:
    lowered = prompt.lower()
    has_context_block = bool(CONTEXT_BLOCK_PATTERN.search(prompt)) or "following" in lowered

    return PromptFeatures(
        word_count=len(prompt.split()),
        simple_keyword_matches=_count_keyword_matches(lowered, SIMPLE_KEYWORDS),
        moderate_keyword_matches=_count_keyword_matches(lowered, MODERATE_KEYWORDS),
        complex_keyword_matches=_count_keyword_matches(lowered, COMPLEX_KEYWORDS),
        constraint_count=sum(
            len(re.findall(pattern, lowered)) for pattern in CONSTRAINT_PATTERNS
        ),
        has_long_context_block=has_context_block,
    )
