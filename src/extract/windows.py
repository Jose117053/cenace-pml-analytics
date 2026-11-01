from datetime import date, timedelta


def generate_date_windows(start_date: str | date, end_date: str | date, max_days: int = 7):
    """
    Split a date range into contiguous sub-periods of up to `max_days` days inclusive.

    CENACE limits API queries to 1 to 7 operating days inclusive per request:
    Example: (2024-01-01, 2024-01-07) comprises exactly 7 days (168 hours).
    The next window starts strictly on the day following the previous end date -> (2024-01-08, 2024-01-14).

    Args:
        start_date: Start date ('YYYY-MM-DD' or datetime.date).
        end_date: End date ('YYYY-MM-DD' or datetime.date).
        max_days: Maximum number of days per window (7 by default).

    Returns:
        List of (start_date, end_date) tuples.
    """
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    if start_date > end_date:
        raise ValueError(f"start_date ({start_date}) must be less than or equal to end_date ({end_date})")

    windows = []
    current_start = start_date
    delta = timedelta(days=max_days - 1)

    while current_start <= end_date:
        current_end = min(current_start + delta, end_date)
        windows.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)

    return windows
