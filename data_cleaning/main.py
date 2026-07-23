import sys
import pandas as pd
import polars as pl
import logging
import time
from pathlib import Path
from src.report import analyze_step_drop, generate_html_report
from src.utils import deduplicate_records, log_row_reduction, init_audit_file, write_audit_step, finalize_audit_report
from src.normalize import normalize_names
from src.chain_removal import filter_chains_and_duplicates
from src.geo_dedup import balltree_spatial_deduplication
from src.adv_filters import apply_hours_and_fuzzy_filters
from src.final_regex import final_keyword_exclusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_pipeline(modules_to_run, save_tracking=True, items_to_track=None):
    start_time = time.time()
    
    # 1. Path definitions
    DATA_DIR = Path("data") 
    INPUT_FILE = DATA_DIR / "raw" / "dataplor_cleaned.csv"
    EXTERNAL_KEYWORDS = DATA_DIR / "external" / "List_keywords_regex - Sheet1.csv"
    OUTPUT_FILE = DATA_DIR / "processed" / "dataplor_final_pipeline_output.csv"

    logging.info(f"Starting pipeline. Modules: {modules_to_run} | Auditing items: {items_to_track}")
    
    # Inicialización de auditorías y reportes
    audit_file = init_audit_file(DATA_DIR, items_to_track)  
    report_metrics = []

    # Load initial data
    df = pd.read_csv(INPUT_FILE, low_memory=False, encoding='utf-8')
    current_df = pl.from_pandas(df)
    initial_data_ref = current_df 
    
    logging.info(f"Initial records: {len(current_df)}")

    # Module 1: Normalization
    if 1 in modules_to_run:
        logging.info("Executing Module 1: Normalization...")
        prev_df = current_df

        current_df = deduplicate_records(
            df=normalize_names(current_df), 
            subset=["name_normalized", "latitude", "longitude"]
        )

        # Registrar métricas para el reporte HTML
        report_metrics.append(analyze_step_drop(prev_df, current_df, "Module 1: Normalization"))

        if save_tracking: current_df.write_csv(DATA_DIR / "processed" / "01_normalized_deduped.csv", include_bom=True)
        log_row_reduction(prev_df, current_df, "Module 1")

    # Module 2: Chain filter
    if 2 in modules_to_run:
        logging.info("Executing Module 2: Chain removal...")
        prev_df = current_df

        current_df, items_to_track, mod_report = filter_chains_and_duplicates(
            df=current_df,
            filter_duplicates=False,
            max_appearances=4,
            items_to_track=items_to_track
        )
        
        if mod_report:
            write_audit_step(audit_file, "MODULE 2 (Chain Filter)", mod_report)

        report_metrics.append(analyze_step_drop(prev_df, current_df, "Module 2: Chain Filter"))

        if save_tracking: current_df.write_csv(DATA_DIR / "processed" / "02_chains_filtered.csv", include_bom=True)
        log_row_reduction(prev_df, current_df, "Module 2")

    # Module 3: Geo Deduplication
    if 3 in modules_to_run:
        logging.info("Executing Module 3: BallTree Deduplication...")
        prev_df = current_df

        current_df, items_to_track, mod_report = balltree_spatial_deduplication(
            current_df,
            radius_meters=50,
            items_to_track=items_to_track
        )

        if mod_report:
            write_audit_step(audit_file, "MODULE 3 (BallTree Filter)", mod_report)

        report_metrics.append(analyze_step_drop(prev_df, current_df, "Module 3: BallTree Dedup"))

        if save_tracking: current_df.write_csv(DATA_DIR / "processed" / "03_geo_balltree_deduped.csv", include_bom=True)
        log_row_reduction(prev_df, current_df, "Module 3")

    # Module 4: Hours and Fuzzy
    if 4 in modules_to_run:
        logging.info("Executing Module 4: Hours and Fuzzy Deduplication...")
        prev_df = current_df

        current_df, items_to_track, mod_report = apply_hours_and_fuzzy_filters(
            current_df,
            min_hours=20,
            night_start=19,
            fuzzy_thresh=80,
            dist_m=100,
            items_to_track=items_to_track
        )
        
        if mod_report:
            write_audit_step(audit_file, "MODULE 4 (Hours and Fuzzy Filter)", mod_report)

        report_metrics.append(analyze_step_drop(prev_df, current_df, "Module 4: Hours & Fuzzy"))

        if save_tracking: current_df.write_csv(DATA_DIR / "processed" / "04_hours_fuzzy_filtered.csv", include_bom=True)
        log_row_reduction(prev_df, current_df, "Module 4")

    # Module 5: Final Regex
    if 5 in modules_to_run:
        logging.info("Executing Module 5: Final exclusion via external regex...") 
        prev_df = current_df
        
        current_df, items_to_track, mod_report = final_keyword_exclusion(
            current_df,
            EXTERNAL_KEYWORDS,
            items_to_track=items_to_track
        )
        
        if mod_report:
            write_audit_step(audit_file, "MODULE 5 (Regex Filter)", mod_report)

        report_metrics.append(analyze_step_drop(prev_df, current_df, "Module 5: Final Regex"))

        if save_tracking: current_df.write_csv(DATA_DIR / "processed" / "05_external_regex_excluded.csv", include_bom=True)
        log_row_reduction(prev_df, current_df, "Module 5")

    # --- CIERRE DE AUDITORÍA Y GENERACIÓN DE REPORTE HTML ---
    finalize_audit_report(audit_file, current_df, items_to_track)
    
    if report_metrics:
        generate_html_report(report_metrics, DATA_DIR)

    # Save final results
    log_row_reduction(initial_data_ref, current_df, "Full Pipeline Process")
    current_df.write_csv(OUTPUT_FILE, include_bom=True)
    
    elapsed = time.time() - start_time
    logging.info(f"Done in {elapsed/60:.2f} mins. Final file saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    # --- 1. Choose which modules to run ---
    modules_to_run = [ 4]
    
    # --- 2. Enable/Disable saving huge intermediate CSVs ---
    save_intermediate_csvs = False 
    
    # --- 3. AUDIT TRACKING LIST ---
    # Put the text you want to track here (e.g., store names, specific words)
    # Leave it as an empty list [] if you don't want to track anything.
    track_these_elements = [
        'ABARROTES'
        ] 
    
    run_pipeline(
        modules_to_run=modules_to_run, 
        save_tracking=save_intermediate_csvs,
        items_to_track=track_these_elements
    )