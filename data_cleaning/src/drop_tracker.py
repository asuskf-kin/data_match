# src/drop_tracker.py
import logging
from pathlib import Path

import polars as pl


def save_dropped_records(
    prev_df: pl.DataFrame,
    current_df: pl.DataFrame,
    data_dir: Path,
    filename: str,
    save_drops: bool = True,
    id_col: str = None,
):
    """
    Compares two DataFrames and saves the rows that were removed into data/dropped.

    Parameters
    ----------
    prev_df : pl.DataFrame
        DataFrame before applying the filter.
    current_df : pl.DataFrame
        DataFrame after applying the filter.
    data_dir : Path
        Base data directory (DATA_DIR).
    filename : str
        Name of the CSV file (should match the intermediate step name).
    save_drops : bool
        Flag to enable or disable saving the dropped records.
    id_col : str, optional
        Unique identifier column (e.g., 'place_id'). If not provided, it compares the entire row.
    """
    # If the option is disabled or no rows were removed, exit early
    if not save_drops or prev_df.height <= current_df.height:
        return

    dropped_dir = data_dir / "dropped"
    dropped_dir.mkdir(parents=True, exist_ok=True)

    try:
        if id_col and id_col in prev_df.columns and id_col in current_df.columns:
            # If a unique ID exists, anti join is much faster and safer
            join_cols = [id_col]
        else:
            # If no ID exists, we find common columns to avoid errors
            # in case the module added new columns (e.g., name_normalized)
            join_cols = [col for col in prev_df.columns if col in current_df.columns]

        # The "anti join" extracts everything from prev_df that DOES NOT exist in current_df
        dropped_df = prev_df.join(
            current_df.select(join_cols), on=join_cols, how="anti"
        )

        if dropped_df.height > 0:
            out_path = dropped_dir / filename
            dropped_df.write_csv(out_path, include_bom=True)
            logging.info(f"💾 Saved {dropped_df.height} dropped records to {out_path}")

    except Exception as e:
        logging.error(f"⚠️ Could not save dropped records for {filename}: {e}")
