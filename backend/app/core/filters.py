from datetime import datetime, timedelta

from sqlalchemy.orm import Query


def apply_date_range(query: Query, column, date_from: str | None, date_to: str | None) -> Query:
    """Filter `query` on `column` (a DateTime column) by inclusive calendar-day
    bounds given as "YYYY-MM-DD" strings."""
    if date_from:
        query = query.filter(column >= datetime.fromisoformat(date_from))
    if date_to:
        query = query.filter(column < datetime.fromisoformat(date_to) + timedelta(days=1))
    return query
