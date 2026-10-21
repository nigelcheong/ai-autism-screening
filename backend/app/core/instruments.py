"""Canonical content for the two age-based screening instruments, extracted
from the standardized tools in frontend/QuestionsData/ (Allison, Auyeung &
Baron-Cohen, 2012, J Am Acad Child Adolesc Psychiatry 51(2):202-12).

Each item is (text, options, scoring_indices): scoring_indices lists which
0-based option indices count as a "concerning" / scorable response for that
item. Raw score = number of items where the selected option index is in that
item's scoring_indices (out of 10); see app/core/age.py for the category
cutoffs each instrument uses.
"""

AQ10_AGREE_SCALE = ["Definitely Agree", "Slightly Agree", "Slightly Disagree", "Definitely Disagree"]
_AGREE_SCORED = [0, 1]  # Definitely/Slightly Agree
_DISAGREE_SCORED = [2, 3]  # Slightly/Definitely Disagree

# AQ-10 (ages 4-11). Cutoff: raw score >= 6 -> consider referral.
AQ10_ITEMS = [
    ("I often notice small sounds when others do not.", AQ10_AGREE_SCALE, _AGREE_SCORED),
    (
        "When I'm reading a story, I find it difficult to work out the characters' intentions.",
        AQ10_AGREE_SCALE,
        _AGREE_SCORED,
    ),
    (
        "I find it easy to \"read between the lines\" when someone is talking to me.",
        AQ10_AGREE_SCALE,
        _DISAGREE_SCORED,
    ),
    (
        "I usually concentrate more on the whole picture, rather than the small details.",
        AQ10_AGREE_SCALE,
        _DISAGREE_SCORED,
    ),
    ("I know how to tell if someone listening to me is getting bored.", AQ10_AGREE_SCALE, _DISAGREE_SCORED),
    ("I find it easy to do more than one thing at once.", AQ10_AGREE_SCALE, _DISAGREE_SCORED),
    (
        "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
        AQ10_AGREE_SCALE,
        _DISAGREE_SCORED,
    ),
    (
        "If there is an interruption, I can switch back to what I was doing very quickly.",
        AQ10_AGREE_SCALE,
        _DISAGREE_SCORED,
    ),
    (
        "I like to collect information about categories of things (e.g. types of car, "
        "types of bird, types of train, types of plant etc).",
        AQ10_AGREE_SCALE,
        _AGREE_SCORED,
    ),
    ("I find it difficult to work out people's intentions.", AQ10_AGREE_SCALE, _AGREE_SCORED),
]

_FREQ_SCALE = ["Many times a day", "A few times a day", "A few times a week", "Less than once a week", "Never"]
_ALWAYS_SCALE = ["Always", "Usually", "Sometimes", "Rarely", "Never"]
_LAST_3 = [2, 3, 4]  # columns C, D, E
_FIRST_3 = [0, 1, 2]  # columns A, B, C (item 10 is reversed)

# Q-CHAT-10 (ages 18-48 months). Cutoff: raw score >= 3 -> consider referral.
QCHAT10_ITEMS = [
    ("Does your child look at you when you call his/her name?", _ALWAYS_SCALE, _LAST_3),
    (
        "How easy is it for you to get eye contact with your child?",
        ["Very easy", "Quite easy", "Quite difficult", "Very difficult", "Impossible"],
        _LAST_3,
    ),
    (
        "Does your child point to indicate that s/he wants something? "
        "(e.g. a toy that is out of reach)",
        _FREQ_SCALE,
        _LAST_3,
    ),
    (
        "Does your child point to share interest with you? "
        "(e.g. pointing at an interesting sight)",
        _FREQ_SCALE,
        _LAST_3,
    ),
    ("Does your child pretend? (e.g. care for dolls, talk on a toy phone)", _FREQ_SCALE, _LAST_3),
    ("Does your child follow where you're looking?", _FREQ_SCALE, _LAST_3),
    (
        "If you or someone else in the family is visibly upset, does your child show signs "
        "of wanting to comfort them? (e.g. stroking hair, hugging them)",
        _ALWAYS_SCALE,
        _LAST_3,
    ),
    (
        "Would you describe your child's first words as:",
        ["Very typical", "Quite typical", "Slightly unusual", "Very unusual", "My child doesn't speak"],
        _LAST_3,
    ),
    ("Does your child use simple gestures? (e.g. wave goodbye)", _FREQ_SCALE, _LAST_3),
    ("Does your child stare at nothing with no apparent purpose?", _FREQ_SCALE, _FIRST_3),
]

INSTRUMENT_LABELS = {
    "aq10": "AQ-10 (ages 4-11)",
    "qchat10": "Q-CHAT-10 (ages 18-48 months)",
}

# Both instruments' clinical cutoffs fall inside this shared tiering:
# AQ-10's "refer at >=6" sits at the low/high boundary; Q-CHAT-10's
# "refer at >=3" sits at the low/moderate boundary.
def category_for_raw_score(raw_score: int) -> str:
    if raw_score >= 6:
        return "high"
    if raw_score >= 3:
        return "moderate"
    return "low"


# Instrument-specific referral labels, shown on the report instead of a bare
# percentage -- each instrument's own clinical cutoff score, worded for a
# clinician deciding what to do next, not just "low/moderate/high". The
# "moderate" tier is deliberately a single exact score for each instrument
# (the point right at the standard referral cutoff), not a range.
_REFERRAL_LABELS = {
    "aq10": [
        (5, "No Referral Indicated", "low"),
        (6, "Specialist Review Recommended", "moderate"),
        (10, "Urgent Specialist Evaluation Advised", "high"),
    ],
    "qchat10": [
        (2, "No Referral Indicated", "low"),
        (3, "Early Intervention Review Recommended", "moderate"),
        (10, "Comprehensive Pediatric Evaluation Advised", "high"),
    ],
}


def referral_label_for_raw_score(instrument: str, raw_score: int) -> tuple[str, str] | None:
    """(label, tone) for this instrument's raw score, tone in {low,moderate,high}
    for the report's colour coding. None for an unrecognised instrument."""
    tiers = _REFERRAL_LABELS.get(instrument)
    if tiers is None:
        return None
    for max_score, label, tone in tiers:
        if raw_score <= max_score:
            return label, tone
    return tiers[-1][1], tiers[-1][2]


def referral_band_legend(instrument: str) -> str | None:
    """Three lines, one per tier, e.g.:
        0-5: No Referral Indicated
        6: Specialist Review Recommended
        7-10: Urgent Specialist Evaluation Advised
    Derived from the same tiers referral_label_for_raw_score uses, so the
    two can never drift apart. None for an unrecognised instrument. Each
    line can run to 30-40 characters, too long to keep on one line
    alongside the other summary stats -- \\n-separated, like BAND_LEGEND,
    for the same whitespace-pre-line (web) / splitTextToSize (PDF)
    multi-line handling.
    """
    tiers = _REFERRAL_LABELS.get(instrument)
    if tiers is None:
        return None
    parts = []
    low = 0
    for max_score, label, _tone in tiers:
        range_text = str(low) if low == max_score else f"{low}-{max_score}"
        parts.append(f"{range_text}: {label}")
        low = max_score + 1
    return "\n".join(parts)
