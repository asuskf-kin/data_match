import gc

from config.settings import load_config
from src.reporter import generate_and_save_reports
from src.utils import load_raw_data, run_diagnostics_pipeline


def main():
    """Executes the complete ETL diagnostic pipeline."""
    # 0. Initial memory cleanup
    print("Initializing pipeline: Forcing garbage collection before starting...")
    gc.collect()

    # Load YAML Configuration
    config = load_config()

    # Calculate strictly required columns to optimize memory and I/O reading from Parquet
    base_cols = config["columns"]["base_columns"]
    key_cols = config["columns"]["key_columns"]
    day_prefixes = config["columns"]["day_prefixes"]

    schedule_cols = [f"{d}_open" for d in day_prefixes] + [
        f"{d}_close" for d in day_prefixes
    ]

    # Deduplicate the list using a set, then convert back to list
    required_columns = list(set(base_cols + key_cols + schedule_cols))

    # 1. Extract (Optimized - Only loads columns strictly needed for diagnostics)
    df = load_raw_data(file_path=config["paths"]["raw_file"], columns=required_columns)

    # 2. Transform (Process Diagnostics)
    flags_df, severe_issues, minor_issues = run_diagnostics_pipeline(
        df=df, config=config
    )

    # 3. Load (Export Parquet and Interactive HTML Report)
    generate_and_save_reports(
        df=flags_df,
        severe_issues=severe_issues,
        minor_issues=minor_issues,
        reports_dir=config["paths"]["reports_dir"],
    )


if __name__ == "__main__":
    main()
