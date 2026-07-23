import re
import polars as pl
from config.patterns_mx import CHAIN_REGEX


def filter_chains_and_duplicates(
    df: pl.DataFrame,
    filter_duplicates: bool = True,
    max_appearances: int = 4,
    items_to_track: list[str] = None,
) -> tuple[pl.DataFrame, list[str] | None, str | None]:
    """Filters commercial chains and duplicate entries from a DataFrame, and updates tracked items.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing a 'name_normalized' column.
    filter_duplicates : bool, default True
        Whether to filter out entries that exceed `max_appearances`.
    max_appearances : int, default 4
        Maximum allowed occurrences of a normalized name before being flagged as a duplicate.
    items_to_track : list[str], optional
        List of specific strings to trace for auditing purposes.

    Returns
    -------
    tuple[pl.DataFrame, list[str] | None, str | None]
        A tuple containing:
        - The filtered/cleaned DataFrame.
        - The updated `items_to_track` list (unmodified if not auditing).
        - An audit report string if `items_to_track` was provided and items were found,
          otherwise None.

    Raises
    ------
    ValueError
        If 'name_normalized' column is missing from the input DataFrame.
    """
    if "name_normalized" not in df.columns:
        raise ValueError(
            "Column 'name_normalized' is missing. Run normalize_names first."
        )

    chain_pattern = "(?i)" + "|".join(f"(?:{p})" for p in CHAIN_REGEX)

    df_clean = df.with_columns(
        pl.col("name_normalized")
        .str.replace_all(r"[´`‘’ʼ]", "'")
        .str.to_lowercase()
        .alias("_filter_key")
    )

    is_auditing = items_to_track is not None and len(items_to_track) > 0

    # ==========================================
    # 🚀 FAST MODE (No auditing -> No list modifications)
    # ==========================================
    if not is_auditing:
        df_clean = df_clean.filter(
            ~pl.col("_filter_key").str.contains(chain_pattern).fill_null(False)
        )
        if filter_duplicates:
            df_clean = df_clean.filter(
                pl.col("_filter_key").is_null()
                | (pl.len().over("_filter_key") <= max_appearances)
            )
        if "identified_as_chain" in df_clean.columns:
            df_clean = df_clean.filter(
                pl.col("identified_as_chain").cast(pl.String).fill_null("")
                != "True"
            )
        # Returns the DataFrame, leaving items_to_track untouched (or None)
        return df_clean.drop("_filter_key"), items_to_track, None

    # ==========================================
    # 🔍 AUDIT MODE (Only applies tracking logic here)
    # ==========================================
    df_clean = df_clean.with_columns(
        pl.lit(None).cast(pl.String).alias("drop_reason")
    )

    # 1. Regex Filter (Chains)
    regex_mask = pl.col("_filter_key").str.contains(chain_pattern).fill_null(False)
    df_clean = df_clean.with_columns(
        pl.when(regex_mask & pl.col("drop_reason").is_null())
        .then(pl.lit("1_Regex_Chain"))
        .otherwise(pl.col("drop_reason"))
        .alias("drop_reason")
    )

    # 2. Duplicates Filter
    if filter_duplicates:
        dup_mask = pl.col("_filter_key").is_not_null() & (
            pl.len().over("_filter_key") > max_appearances
        )
        df_clean = df_clean.with_columns(
            pl.when(dup_mask & pl.col("drop_reason").is_null())
            .then(pl.lit(f"2_Too_Many_Duplicates_(>{max_appearances})"))
            .otherwise(pl.col("drop_reason"))
            .alias("drop_reason")
        )

    # 3. Source Flag Filter
    if "identified_as_chain" in df_clean.columns:
        flag_mask = (
            pl.col("identified_as_chain").cast(pl.String).fill_null("") == "True"
        )
        df_clean = df_clean.with_columns(
            pl.when(flag_mask & pl.col("drop_reason").is_null())
            .then(pl.lit("3_Flagged_As_Chain_In_Source"))
            .otherwise(pl.col("drop_reason"))
            .alias("drop_reason")
        )

    # Separate survivors and dropped records
    df_dropped = df_clean.filter(pl.col("drop_reason").is_not_null()).drop(
        "_filter_key"
    )
    df_survivors = df_clean.filter(pl.col("drop_reason").is_null()).drop(
        ["_filter_key", "drop_reason"]
    )

    # --- BUILD AUDIT REPORT & UPDATE TRACKED ITEMS ---
    report_lines = []
    dropped_items_set = set()

    for text in items_to_track:
        text_lower = text.lower()
        survived = df_survivors.filter(
            pl.col("name_normalized")
            .str.to_lowercase()
            .str.contains(text_lower, literal=True)
        )
        dropped = df_dropped.filter(
            pl.col("name_normalized")
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
            # Mark item as dropped to remove it from items_to_track
            dropped_items_set.add(text)
            reasons = dropped.group_by("drop_reason").agg(
                pl.len().alias("count")
            )
            for row in reasons.iter_rows():
                report_lines.append(
                    f"    ❌ DROPPED: Removed at this step due to: [{row[0]}] ({row[1]} records dropped)."
                )

    # Filter out dropped elements from the original items_to_track list
    updated_items_to_track = [
        item for item in items_to_track if item not in dropped_items_set
    ]

    if len(report_lines) == 0:
        return df_survivors, updated_items_to_track, None

    final_report_text = (
        "=" * 45
        + "\n🎯 AUDIT: MODULE 2 (Chains)\n"
        + "=" * 45
        + "".join(report_lines)
        + "\n"
    )

    return df_survivors, updated_items_to_track, final_report_text