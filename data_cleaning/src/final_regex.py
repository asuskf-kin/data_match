# src/05_final_regex.py
import re
from pathlib import Path
import polars as pl
from config.keep_words import KEEP_WORDS


def final_keyword_exclusion(
    df: pl.DataFrame,
    keywords_csv_path: str,
    items_to_track: list[str] = None,
) -> tuple[pl.DataFrame, list[str] | None, str | None]:
    """Filters businesses based on an external keywords CSV file, using Regular Expressions while respecting a whitelist (KEEP_WORDS).

    Parameters
    ----------
    df : pl.DataFrame
        The input Polars DataFrame containing business data (must have a 'name' column).
    keywords_csv_path : str
        The file path to the CSV containing the exclusion keywords.
    items_to_track : list[str], optional
        List of specific strings to trace for auditing purposes.

    Returns
    -------
    tuple[pl.DataFrame, list[str] | None, str | None]
        A tuple containing:
        - The clean DataFrame (survivors).
        - The updated `items_to_track` list (unmodified if not auditing).
        - An audit report string if `items_to_track` was provided and items were processed,
          otherwise None.
    """
    is_auditing = items_to_track is not None and len(items_to_track) > 0

    # 1. Read external keywords
    try:
        df_keywords = pl.read_csv(keywords_csv_path, infer_schema_length=0)
        col_name = df_keywords.columns[0]
        keywords = df_keywords.get_column(col_name).drop_nulls().to_list()
    except Exception as e:
        print(
            f"Error reading {keywords_csv_path}: {e}. Returning unfiltered dataset."
        )
        return df, items_to_track, None

    if df.is_empty() or not keywords:
        return df, items_to_track, None

    # 2. Regex transformation helper functions
    def contains_special_characters(kw_list):
        return any(re.search(r"[\[\]().?*+^$|]", kw) for kw in kw_list)

    def remove_accents(text):
        return text.translate(str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU"))

    def transform_keywords_to_regex(kw_list):
        tilde_map = {
            "a": "[aá]",
            "e": "[eé]",
            "i": "[ií]",
            "o": "[oó]",
            "u": "[uú]",
            "A": "[AÁ]",
            "E": "[EÉ]",
            "I": "[IÍ]",
            "O": "[OÓ]",
            "U": "[UÚ]",
        }
        trf = []
        for kw in kw_list:
            for letter, pat in tilde_map.items():
                kw = kw.replace(letter, pat)
            esc = re.escape(kw).replace("\\ ", r"\s*").replace(" ", r"\s*")
            if len(kw) == 4:
                trf.append(r"\b" + esc + r"\b")
            else:
                base = r"\b" + esc + r"\w*"
                trf.append(base)
                if not kw.endswith("s"):
                    trf.append(base + r"s?")
        return trf

    def detect_keyword(text, kw_list):
        if not isinstance(text, str):
            return None
        text_lower = text.lower()
        for kw in kw_list:
            if kw.lower() in text_lower:
                return kw
        return None

    kws = keywords + [
        remove_accents(kw) for kw in keywords if remove_accents(kw) != kw
    ]
    use_regex = contains_special_characters(kws)

    if not use_regex:
        patterns = transform_keywords_to_regex(kws)
    else:
        patterns = [
            r"\b"
            + re.sub(r"\(([^)]+)\)", r"(?:\1)", kw)
            .replace("\\ ", r"\s*")
            .replace(" ", r"\s*")
            + r"\w*"
            for kw in kws
        ]

    def build_chunked_regex_mask(
        col_name: str, pattern_list: list, chunk_size: int = 150
    ):
        if not pattern_list:
            return pl.lit(False)
        exprs = []
        for i in range(0, len(pattern_list), chunk_size):
            chunk = pattern_list[i : i + chunk_size]
            regex_str = "(?i)" + "|".join(chunk)
            exprs.append(
                pl.col(col_name).str.contains(regex_str).fill_null(False)
            )
        return pl.any_horizontal(exprs)

    # 3. Apply Regex and mask exceptions (using chunks)
    df_clean = df.with_row_index("__row_id")

    mask_regex = build_chunked_regex_mask("name", patterns, chunk_size=150)
    keep_patterns = [re.escape(w) for w in KEEP_WORDS]
    mask_keep = build_chunked_regex_mask("name", keep_patterns, chunk_size=150)

    # Filter candidates for removal
    df_temp = df_clean.filter(mask_regex & ~mask_keep)

    # Detect exact keyword responsible
    df_temp = df_temp.with_columns(
        pl.col("name")
        .map_elements(
            lambda x: detect_keyword(x, keywords), return_dtype=pl.String
        )
        .alias("detected_keyword")
    )

    df_filtered_out_full = df_temp.filter(
        pl.col("detected_keyword").is_not_null()
    )
    excluded_ids = df_filtered_out_full.get_column("__row_id")

    # Flag drop reasons for the audit step
    df_clean = df_clean.with_columns(
        pl.when(pl.col("__row_id").is_in(excluded_ids))
        .then(
            pl.concat_str(
                [
                    pl.lit("1_Final_Regex_Keyword_("),
                    pl.col("name").map_elements(
                        lambda x: detect_keyword(x, keywords) or "Unknown",
                        return_dtype=pl.String,
                    ),
                    pl.lit(")"),
                ]
            )
        )
        .otherwise(pl.lit(None))
        .alias("drop_reason")
    )

    # ==========================================
    # 🚀 FAST MODE (No auditing)
    # ==========================================
    if not is_auditing:
        df_survivors = df_clean.filter(
            pl.col("drop_reason").is_null()
        ).drop(["__row_id", "drop_reason"], strict=False)
        return df_survivors, items_to_track, None

    # ==========================================
    # 🔍 AUDIT MODE
    # ==========================================
    df_dropped = df_clean.filter(pl.col("drop_reason").is_not_null()).drop(
        "__row_id", strict=False
    )
    df_survivors = df_clean.filter(pl.col("drop_reason").is_null()).drop(
        ["__row_id", "drop_reason"], strict=False
    )

    # --- BUILD AUDIT REPORT & UPDATE TRACKED ITEMS ---
    report_lines = []
    dropped_items_set = set()

    for text in items_to_track:
        text_lower = text.lower()
        survived = df_survivors.filter(
            pl.col("name")
            .str.to_lowercase()
            .str.contains(text_lower, literal=True)
        )
        dropped = df_dropped.filter(
            pl.col("name")
            .str.to_lowercase()
            .str.contains(text_lower, literal=True)
        )

        if len(survived) == 0 and len(dropped) == 0:
            continue

        report_lines.append(f"\n🔍 Tracked text: '{text}'")

        if len(survived) > 0:
            report_lines.append(
                f"    ✅ SURVIVED: Passed this filter successfully ({len(survived)} records)."
            )

        if len(dropped) > 0:
            dropped_items_set.add(text)
            reasons = dropped.group_by("drop_reason").agg(
                pl.len().alias("count")
            )
            for row in reasons.iter_rows():
                report_lines.append(
                    f"    ❌ DROPPED: Removed at this step due to: [{row[0]}] ({row[1]} records dropped)."
                )

    # Prune dropped items from tracked list
    updated_items_to_track = [
        item for item in items_to_track if item not in dropped_items_set
    ]

    if len(report_lines) == 0:
        return df_survivors, updated_items_to_track, None

    final_report_text = (
        "=" * 45
        + "\n🎯 AUDIT: MODULE 5 (Final Regex Keyword Exclusion)\n"
        + "=" * 45
        + "".join(report_lines)
        + "\n"
    )

    return df_survivors, updated_items_to_track, final_report_text