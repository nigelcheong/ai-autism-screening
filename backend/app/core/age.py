"""Age-based screening instrument selection.

Two instruments are supported, matching the standardized tools in
frontend/QuestionsData/:
  - "qchat10": Q-CHAT-10, for toddlers 18-48 months (Allison et al. 2012)
  - "aq10":    AQ-10, for children 4-11 years (4-point agree/disagree scale)

Age 48 months (4 years) is the cutoff between them.
"""

from datetime import date, datetime

QCHAT_TO_AQ10_CUTOFF_MONTHS = 48

# Image screening covers toddlers through pre-teens: 18 months to 11 years
# (inclusive) -- i.e. below 12 years.
IMAGE_SCREENING_MIN_MONTHS = 18
IMAGE_SCREENING_MAX_MONTHS = 144


def age_in_months(dob: date, as_of: date | None = None) -> int:
    as_of = as_of or date.today()
    months = (as_of.year - dob.year) * 12 + (as_of.month - dob.month)
    if as_of.day < dob.day:
        months -= 1
    return max(months, 0)


def instrument_for_age_months(months: int) -> str:
    return "qchat10" if months < QCHAT_TO_AQ10_CUTOFF_MONTHS else "aq10"


def is_eligible_for_image_screening(months: int) -> bool:
    return IMAGE_SCREENING_MIN_MONTHS <= months < IMAGE_SCREENING_MAX_MONTHS


def instrument_for_dob(dob_str: str) -> str:
    dob = datetime.fromisoformat(dob_str).date()
    return instrument_for_age_months(age_in_months(dob))
