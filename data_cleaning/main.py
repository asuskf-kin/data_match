# main.py
import logging
import time

import pandas as pd
import polars as pl

from config.settings import ACTIVE_COUNTRY, DATA_DIR, INPUT_FILE, OUTPUT_FILE
from src.adv_filters import apply_hours_and_fuzzy_filters
from src.chain_removal import filter_chains_and_duplicates
from src.drop_tracker import save_dropped_records  # <--- NEW IMPORT
from src.final_regex import final_keyword_exclusion
from src.geo_dedup import balltree_spatial_deduplication
from src.normalize import normalize_names
from src.report import analyze_step_drop, generate_html_report
from src.utils import (
    deduplicate_records,
    finalize_audit_report,
    init_audit_file,
    log_row_reduction,
    write_audit_step,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_pipeline(
    modules_to_run, save_tracking=True, save_drops=True, items_to_track=None
):  # <--- ADDED save_drops
    start_time = time.time()

    logging.info(
        f"Starting pipeline. Modules: {modules_to_run} | Auditing items: {items_to_track}"
    )
    print("=" * 50)
    print(f"🚀 STARTING PIPELINE FOR COUNTRY: {ACTIVE_COUNTRY.upper()}")
    print("=" * 50)

    print(f"📁 Input file: {INPUT_FILE}")
    print(f"📁 Output file: {OUTPUT_FILE}\n")

    audit_file = init_audit_file(DATA_DIR, items_to_track)
    report_metrics = []

    df = pd.read_csv(INPUT_FILE, low_memory=False, encoding="utf-8")
    current_df = pl.from_pandas(df)
    initial_data_ref = current_df

    logging.info(f"Initial records: {len(current_df)}")

    # ==========================================
    # Module 1: Normalization
    # ==========================================
    if 1 in modules_to_run:
        logging.info("Executing Module 1: Normalization...")
        prev_df = current_df

        current_df = deduplicate_records(
            df=normalize_names(current_df),
            subset=["name_normalized", "latitude", "longitude"],
        )

        report_metrics.append(
            analyze_step_drop(prev_df, current_df, "Module 1: Normalization")
        )
        filename_mod1 = "01_normalized_deduped.csv"

        if save_tracking:
            current_df.write_csv(
                DATA_DIR / "processed" / filename_mod1, include_bom=True
            )

        # NEW LINE: Save dropped records
        save_dropped_records(prev_df, current_df, DATA_DIR, filename_mod1, save_drops)

        log_row_reduction(prev_df, current_df, "Module 1")

    # ==========================================
    # Module 2: Chain filter
    # ==========================================
    if 2 in modules_to_run:
        logging.info("Executing Module 2: Chain removal...")
        prev_df = current_df

        current_df, items_to_track, mod_report = filter_chains_and_duplicates(
            df=current_df,
            filter_duplicates=False,
            max_appearances=4,
            items_to_track=items_to_track,
        )

        if mod_report:
            write_audit_step(audit_file, "MODULE 2 (Chain Filter)", mod_report)

        report_metrics.append(
            analyze_step_drop(prev_df, current_df, "Module 2: Chain Filter")
        )
        filename_mod2 = "02_chains_filtered.csv"

        if save_tracking:
            current_df.write_csv(
                DATA_DIR / "processed" / filename_mod2, include_bom=True
            )

        # NEW LINE: Save dropped records
        save_dropped_records(prev_df, current_df, DATA_DIR, filename_mod2, save_drops)

        log_row_reduction(prev_df, current_df, "Module 2")

    # ==========================================
    # Module 3: Geo Deduplication
    # ==========================================
    if 3 in modules_to_run:
        logging.info("Executing Module 3: BallTree Deduplication...")
        prev_df = current_df

        current_df, items_to_track, mod_report = balltree_spatial_deduplication(
            current_df, radius_meters=50, items_to_track=items_to_track
        )

        if mod_report:
            write_audit_step(audit_file, "MODULE 3 (BallTree Filter)", mod_report)

        report_metrics.append(
            analyze_step_drop(prev_df, current_df, "Module 3: BallTree Dedup")
        )
        filename_mod3 = "03_geo_balltree_deduped.csv"

        if save_tracking:
            current_df.write_csv(
                DATA_DIR / "processed" / filename_mod3, include_bom=True
            )

        # NEW LINE: Save dropped records
        save_dropped_records(prev_df, current_df, DATA_DIR, filename_mod3, save_drops)

        log_row_reduction(prev_df, current_df, "Module 3")

    # ==========================================
    # Module 4: Hours and Fuzzy
    # ==========================================
    if 4 in modules_to_run:
        logging.info("Executing Module 4: Hours and Fuzzy Deduplication...")
        prev_df = current_df

        current_df, items_to_track, mod_report = apply_hours_and_fuzzy_filters(
            current_df,
            min_hours=20,
            night_start=19,
            fuzzy_thresh=80,
            dist_m=100,
            items_to_track=items_to_track,
        )

        if mod_report:
            write_audit_step(
                audit_file, "MODULE 4 (Hours and Fuzzy Filter)", mod_report
            )

        report_metrics.append(
            analyze_step_drop(prev_df, current_df, "Module 4: Hours & Fuzzy")
        )
        filename_mod4 = "04_hours_fuzzy_filtered.csv"

        if save_tracking:
            current_df.write_csv(
                DATA_DIR / "processed" / filename_mod4, include_bom=True
            )

        # NEW LINE: Save dropped records
        save_dropped_records(prev_df, current_df, DATA_DIR, filename_mod4, save_drops)

        log_row_reduction(prev_df, current_df, "Module 4")

    # ==========================================
    # Module 5: Final Regex
    # ==========================================
    if 5 in modules_to_run:
        logging.info("Executing Module 5: Final exclusion via external regex...")
        prev_df = current_df

        current_df, items_to_track, mod_report = final_keyword_exclusion(
            current_df, items_to_track=items_to_track
        )

        if mod_report:
            write_audit_step(audit_file, "MODULE 5 (Regex Filter)", mod_report)

        report_metrics.append(
            analyze_step_drop(prev_df, current_df, "Module 5: Final Regex")
        )
        filename_mod5 = "05_external_regex_excluded.csv"

        if save_tracking:
            current_df.write_csv(
                DATA_DIR / "processed" / filename_mod5, include_bom=True
            )

        # NEW LINE: Save dropped records
        save_dropped_records(prev_df, current_df, DATA_DIR, filename_mod5, save_drops)

        log_row_reduction(prev_df, current_df, "Module 5")

    # --- CLOSING & REPORTS ---
    finalize_audit_report(audit_file, current_df, items_to_track)

    if report_metrics:
        generate_html_report(report_metrics, DATA_DIR)

    log_row_reduction(initial_data_ref, current_df, "Full Pipeline Process")
    current_df.write_csv(OUTPUT_FILE, include_bom=True)

    elapsed = time.time() - start_time
    logging.info(f"Done in {elapsed / 60:.2f} mins. Final file saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    modules_to_run = [1, 2, 3, 4, 5]

    # --- BOOLEANS TO CONTROL THE FLOW ---
    save_intermediate_csvs = False
    save_dropped_csvs = True

    track_these_elements = []

    run_pipeline(
        modules_to_run=modules_to_run,
        save_tracking=save_intermediate_csvs,
        save_drops=save_dropped_csvs,
        items_to_track=track_these_elements,
    )
