import logging
from pathlib import Path
import time
from typing import List, Optional, Literal
import polars as pl

logger = logging.getLogger(__name__)

def deduplicate_records(
    df: pl.DataFrame, 
    subset: Optional[List[str]] = None,
    keep: Literal["first", "last", "any", "none"] = "first"
) -> pl.DataFrame:
    """
    Removes duplicate rows from a Polars DataFrame and logs the exact 
    number of rows before and after, alongside the reduction percentage.
    
    Args:
        df (pl.DataFrame): The input DataFrame.
        subset (Optional[List[str]]): Columns to consider for identifying duplicates. 
                                      If None, evaluates all columns.
        keep (str): Which duplicate to keep ('first', 'last', 'any', or 'none').
        
    Returns:
        pl.DataFrame: A new deduplicated DataFrame.
    """
    initial_count = df.height
    
    if initial_count == 0:
        logger.warning("The DataFrame is empty. Skipping deduplication.")
        return df

    # Validate subset columns exist to prevent cryptic ComputeErrors
    if subset is not None:
        missing_cols = [col for col in subset if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame for deduplication: {missing_cols}")

    # maintain_order=True keeps the original row sequence
    df_dedup = df.unique(subset=subset, keep=keep, maintain_order=True)
    
    final_count = df_dedup.height
    removed_count = initial_count - final_count
    
    if removed_count > 0:
        reduction_pct = (removed_count / initial_count) * 100
        msg = (
            f"Deduplication: Removed {removed_count} rows. "
            f"Dataset reduced by {reduction_pct:.2f}%. "
            f"Original count: {initial_count} | New count: {final_count} rows."
        )
        logger.info(msg)
    else:
        logger.info(f"Deduplication: No duplicates found. Total rows remains {initial_count}.")
        
    return df_dedup

def log_row_reduction(
    df_before: pl.DataFrame, 
    df_after: pl.DataFrame, 
    step_name: str = "Process"
) -> None:
    """
    Simply compares the height (rows) of two Polars DataFrames 
    and logs the exact reduction metrics.
    """
    initial_count = df_before.height
    final_count = df_after.height
    removed_count = initial_count - final_count
    
    # Evita el error de división por cero por si el df original viene vacío
    reduction_pct = (removed_count / initial_count * 100) if initial_count > 0 else 0
    
    logger.info(
        f"[{step_name}] Removed {removed_count} rows. "
        f"Dataset reduced by {reduction_pct:.2f}%. "
        f"Original count: {initial_count} | New count: {final_count} rows."
    )

def init_audit_file(data_dir: Path, items_to_track: list[str]) -> Path | None:
    """
    Creates the audit directory and the base file if there are items to track.
    Returns the file path or None if auditing is disabled.
    """
    if not items_to_track:
        return None
        
    audit_dir = data_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    audit_file = audit_dir / f"audit_report_{timestamp}.txt"
    
    with open(audit_file, "w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write(f"🚀 AUDIT REPORT STARTED: {timestamp}\n")
        f.write(f"🎯 Tracked items: {items_to_track}\n")
        f.write("====================================================\n\n")
        
    logging.info(f"Audit activated. File created at: {audit_file}")
    return audit_file

def write_audit_step(audit_file: Path, module_name: str, report_text: str):
    """
    Appends the report of a specific module to the audit file.
    """
    if audit_file and report_text:
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- {module_name} ---\n{report_text}\n")

def finalize_audit_report(audit_file: Path, final_df: pl.DataFrame, items_to_track: list[str]):
    """
    Evaluates the final DataFrame against the tracked items and writes the verdict.
    """
    if not audit_file or not items_to_track:
        return
        
    with open(audit_file, "a", encoding="utf-8") as f:
        f.write("\n====================================================\n")
        f.write("🏁 FINAL SUMMARY: DID IT REACH THE END OR WAS IT DROPPED?\n")
        f.write("====================================================\n")
        
        for text in items_to_track:
            # Filter the final DF to see if the item survived
            survivors = final_df.filter(
                pl.col("name_normalized").str.to_lowercase().str.contains(text.lower(), literal=True)
            )
            
            if len(survivors) > 0:
                f.write(f"✅ '{text}': REACHED THE END ({len(survivors)} intact records).\n")
            else:
                f.write(f"❌ '{text}': COMPLETELY DROPPED in some pipeline step.\n")
                
    logging.info("Final audit summary saved successfully.")