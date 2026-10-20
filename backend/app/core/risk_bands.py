"""Single source of truth for mapping a model's predicted probability to the
low/moderate/high category shown throughout the app (risk badges, image
risk_category, report legends). Keeping the boundaries here means a report's
explanatory text can never drift out of sync with the actual badge/category
a session was given.
"""

LOW_MAX = 0.34
MODERATE_MAX = 0.67

_RANGE_LABELS = {
    "low": "under 34%",
    "moderate": "34-66%",
    "high": "67% and above",
}

_MEANINGS = {
    "low": "few or no traits indicative of autism were identified",
    "moderate": "some indicative traits were identified; clinical judgement is advised",
    "high": "most consistent with autism-related traits; clinical referral is recommended",
}

BAND_LEGEND = "Low: under 34%  ·  Moderate: 34-66%  ·  High: 67% and above"


def categorize(score: float) -> str:
    if score < LOW_MAX:
        return "low"
    if score < MODERATE_MAX:
        return "moderate"
    return "high"


def range_label(category: str) -> str:
    return _RANGE_LABELS[category]


def meaning(category: str) -> str:
    return _MEANINGS[category]


def short_label(score: float) -> str:
    """e.g. '89% (High)' -- for contexts where the full band legend is shown nearby."""
    category = categorize(score)
    return f"{score * 100:.0f}% ({category.title()})"


def describe_probability(score: float) -> str:
    """e.g. '72% -- High range (67% and above): most consistent with autism-related traits'."""
    category = categorize(score)
    return (
        f"{score * 100:.0f}% -- {category.title()} range ({range_label(category)}): "
        f"{meaning(category)}"
    )
