"""
String normalization utilities.

This module handles text standardizations required for identifying
exact duplicates and cleaning unstructured text fields.
"""

import unicodedata


def norm_name(x):
    """
    Normalizes a string by removing accents, converting to uppercase,
    and stripping leading/trailing whitespace.

    This function is primarily used to prepare venue names for exact
    duplicate matching in the diagnostic pipeline[cite: 2].

    Args:
        x (str): The raw input string to be normalized.

    Returns:
        str: The fully normalized string. If the input is not a string,
        it returns the original input without modifications[cite: 2].
    """
    if not isinstance(x, str):
        return x

    # Decompose unicode characters to isolate and remove accents
    s = "".join(
        c for c in unicodedata.normalize("NFD", x) if unicodedata.category(c) != "Mn"
    )
    return s.upper().strip()
