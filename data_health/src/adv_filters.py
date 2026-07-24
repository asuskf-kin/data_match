def parse_hours_duration(h_str, bad_hours):
    """
    Calculates the total duration in hours from a given time range string.

    Expects a string in the format 'HH:MM-HH:MM' representing the opening
    and closing times. It correctly handles overnight shifts that cross midnight[cite: 2].

    Args:
        h_str (str): The time range string (e.g., '09:00-17:30').
        bad_hours (set): A set of strings considered invalid or empty (e.g., 'Not available').

    Returns:
        float | None: The calculated duration in hours, or None if the input
        is invalid, missing, or explicitly marked as a bad hour[cite: 2].
    """
    if h_str is None or str(h_str).strip() in bad_hours:
        return None
    try:
        parts = str(h_str).strip().split("-")
        if len(parts) != 2:
            return None

        def to_float(t):
            h, m = t.strip().split(":")
            return int(h) + int(m) / 60

        o, c = to_float(parts[0]), to_float(parts[1])
        duration = (24 - o) + c if c < o else c - o
        return duration if duration > 0 else None
    except Exception:
        return None


def opening_hour(h_str, bad_hours):
    """
    Extracts the opening hour from a time range string as a float.

    Args:
        h_str (str): The time range string (e.g., '19:00-02:00').
        bad_hours (set): A set of strings considered invalid or empty.

    Returns:
        float | None: The opening time converted to a float (e.g., 19.0),
        or None if the string is invalid or unparseable[cite: 2].
    """
    if h_str is None or str(h_str).strip() in bad_hours:
        return None
    try:
        parts = str(h_str).strip().split("-")
        if len(parts) != 2:
            return None

        h, m = parts[0].strip().split(":")
        return int(h) + int(m) / 60
    except Exception:
        return None


def parse_split_hours_duration(open_str, close_str, bad_hours):
    """
    Calculates the total duration in hours from separate open and close time strings.
    """
    if open_str is None or close_str is None:
        return None
    if str(open_str).strip() in bad_hours or str(close_str).strip() in bad_hours:
        return None
    try:

        def to_float(t):
            h, m = t.strip().split(":")
            return int(h) + int(m) / 60

        o = to_float(open_str)
        c = to_float(close_str)
        duration = (24 - o) + c if c < o else c - o
        return duration if duration > 0 else None
    except Exception:
        return None


def opening_hour_split(open_str, bad_hours):
    """
    Extracts the opening hour from a standalone string as a float.
    """
    if open_str is None or str(open_str).strip() in bad_hours:
        return None
    try:
        h, m = str(open_str).strip().split(":")
        return int(h) + int(m) / 60
    except Exception:
        return None
