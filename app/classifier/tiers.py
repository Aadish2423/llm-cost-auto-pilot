from enum import Enum


class ComplexityTier(str, Enum):
    """Tier 1: reformatting, extraction, basic Q&A from provided context.
    Tier 2: summarization, classification, structured analysis.
    Tier 3: multi-step reasoning, creative generation, nuanced judgment.

    Values match the keys in config/routing_config.yaml exactly.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
