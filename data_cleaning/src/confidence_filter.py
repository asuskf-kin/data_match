# src/02_confidence_filter.py
import polars as pl

# Drop outlets whose open/closed status confidence is below this value.
OPEN_CLOSED_CONFIDENCE_THRESHOLD = 0.25


def filter_low_open_closed_confidence(
    dataplor_rawdata: pl.DataFrame,
    bottler_cleaning_folder: str,
    container_name_dest: str,
    logger,
    threshold: float = OPEN_CLOSED_CONFIDENCE_THRESHOLD,
    target_audit: str = None  # Added audit capability consistent with the pipeline
) -> pl.DataFrame:
    """
    Removes rows whose open_closed_status_confidence_score is less than `threshold`.
    Rows with missing/non-numeric score are KEPT (they are not "less than" the
    threshold). Does nothing if the column does not exist.
    """
    column = "open_closed_status_confidence_score"

    if column not in dataplor_rawdata.columns:
        logger.warning(f"{column} column not found; skipping open/closed confidence filter.")
        return dataplor_rawdata

    if target_audit:
        print(f"\n{'='*50}\n🔍 AUDITING: '{target_audit}' IN OPEN/CLOSED CONFIDENCE FILTER\n{'='*50}")

    df = dataplor_rawdata.with_row_index("__row_id")

    # Cast column to Float64 (non-numeric values safely become null)
    df_with_scores = df.with_columns(
        pl.col(column).cast(pl.Float64, strict=False).alias("_score_numeric")
    )

    # Condition: Drop ONLY when numeric score is strictly less than threshold
    drop_mask = pl.col("_score_numeric").is_not_null() & (pl.col("_score_numeric") < threshold)

    if target_audit:
        target_rows = df_with_scores.filter(
            pl.col("name").str.to_lowercase().str.count_matches(target_audit.lower(), literal=True).fill_null(0) > 0
        )
        if not target_rows.is_empty():
            target_dropped = target_rows.filter(drop_mask)
            score_val = target_rows.get_column(column).to_list()[0]
            if not target_dropped.is_empty():
                print(f"⚠️ TARGET REMOVED: '{target_audit}' had confidence score '{score_val}' (below threshold {threshold}).")
            else:
                print(f"✅ TARGET KEPT: '{target_audit}' survived (confidence score: '{score_val}').")
        else:
            print(f"STEP 0: Target '{target_audit}' not found in incoming dataset.")
        print(f"{'='*50}\n")

    rows_before = df.height
    df_clean = df_with_scores.filter(~drop_mask).drop(["__row_id", "_score_numeric"])

    logger.info(
        f"Rows filtered for low open/closed confidence (< {threshold}): "
        f"{rows_before} -> {df_clean.height}"
    )

    return df_clean