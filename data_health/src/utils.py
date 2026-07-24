import gc
import time
from pathlib import Path

import polars as pl

from src import diagnostics
from src.adv_filters import opening_hour_split, parse_split_hours_duration
from src.normalize import norm_name

MINOR_ISSUES = ["few_hours", "night_only", "low_quality"]


def load_raw_data(file_path, columns=None):
    """
    Loads the raw Parquet dataset into a Polars DataFrame, selecting only specific columns.
    """
    print(f"Loading file: {file_path}...")
    try:
        if columns:
            print(
                f" -> Optimization: Selectively loading only {len(columns)} necessary columns..."
            )
            # Polars native column selection during read is highly optimized for Parquet
            df = pl.read_parquet(file_path, columns=columns)
        else:
            df = pl.read_parquet(file_path)

        print(f"Records: {df.height:,} | Columns Loaded: {df.width}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise


def run_diagnostics_pipeline(df: pl.DataFrame, config: dict):
    """
    Executes diagnostic checks sequentially, saving intermediate states to disk
    and freeing memory after each step to handle massive datasets safely.
    """
    print("Running diagnostics...")
    start_time = time.time()

    # Establish a primary key for joining intermediate files
    pk = "dataplor_id" if "dataplor_id" in df.columns else "row_id"
    if pk == "row_id":
        df = df.with_row_index("row_id")

    # Set up temporary directory for checkpointing
    reports_dir = Path(config["paths"]["reports_dir"])
    temp_dir = reports_dir / "temp_checkpoints"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(" -> Unpacking configuration variables...")
        day_prefixes = config["columns"]["day_prefixes"]
        key_cols = config["columns"]["key_columns"]
        bad_hours = config["constants"]["bad_hours"]
        exclude_keywords = config["exclude_keywords"]
        min_weekly_hours = config["parameters"]["min_weekly_hours"]
        night_start = config["parameters"]["night_start"]
        low_quality_cut = config["parameters"]["low_quality_cut"]
        geojson_path = config["paths"]["geojson_boundaries"]

        # ---------------------------------------------------------
        # STEP 1: Base Column Expressions
        # ---------------------------------------------------------
        print(" -> [Step 1/5] Evaluating base column expressions...")
        exprs = []
        exprs.extend(diagnostics.get_null_exprs(key_cols))
        exprs.extend(diagnostics.get_geo_exprs())
        exprs.append(diagnostics.get_keyword_expr(exclude_keywords))
        exprs.append(diagnostics.get_quality_expr(df.columns, low_quality_cut))

        step1_df = df.select([pk] + exprs)
        step1_path = temp_dir / "step1_base.parquet"
        step1_df.write_parquet(step1_path)

        print("    [Saved] Step 1 written to disk. Freeing memory...")
        del step1_df, exprs
        gc.collect()

        # ---------------------------------------------------------
        # STEP 2: GeoJSON Spatial Outliers
        # ---------------------------------------------------------
        print(" -> [Step 2/5] Running GeoJSON spatial outlier check...")
        geo_outlier_series = diagnostics.get_geo_outlier_series(df, geojson_path)

        step2_df = df.select([pk]).with_columns(geo_outlier_series)
        step2_path = temp_dir / "step2_geo.parquet"
        step2_df.write_parquet(step2_path)

        print("    [Saved] Step 2 written to disk. Freeing memory...")
        del step2_df, geo_outlier_series
        gc.collect()

        # ---------------------------------------------------------
        # STEP 3: Exact Duplicates
        # ---------------------------------------------------------
        print(" -> [Step 3/5] Identifying exact duplicates...")
        step3_df = (
            df.select(
                [
                    pk,
                    pl.col("name")
                    .map_elements(norm_name, return_dtype=pl.String)
                    .alias("_norm_name"),
                    "latitude",
                    "longitude",
                ]
            )
            .with_columns(
                pl.struct(["_norm_name", "latitude", "longitude"])
                .is_duplicated()
                .alias("exact_duplicate")
            )
            .select([pk, "exact_duplicate"])
        )

        step3_path = temp_dir / "step3_duplicates.parquet"
        step3_df.write_parquet(step3_path)

        print("    [Saved] Step 3 written to disk. Freeing memory...")
        del step3_df
        gc.collect()

        # ---------------------------------------------------------
        # STEP 4: Schedules
        # ---------------------------------------------------------
        print(" -> [Step 4/5] Processing complex schedules and active hours...")

        # FACTORY FUNCTIONS: These guarantee that the loop variable 'd'
        # is strictly isolated and frozen for each day, preventing KeyError.
        def make_dur_mapper(day_name, bad_h):
            def mapper(x):
                return parse_split_hours_duration(
                    x[f"{day_name}_open"], x[f"{day_name}_close"], bad_h
                )

            return mapper

        def make_open_mapper(bad_h):
            def mapper(x):
                return opening_hour_split(x, bad_h)

            return mapper

        # Apply the factory functions to build the expressions safely
        dur_exprs = [
            pl.struct([f"{d}_open", f"{d}_close"])
            .map_elements(make_dur_mapper(d, bad_hours), return_dtype=pl.Float64)
            .alias(f"{d}_dur")
            for d in day_prefixes
        ]

        open_exprs = [
            pl.col(f"{d}_open")
            .map_elements(make_open_mapper(bad_hours), return_dtype=pl.Float64)
            .alias(f"{d}_open_float")
            for d in day_prefixes
        ]

        # Isolate only the columns needed to calculate schedules to save RAM
        sched_cols = (
            [pk]
            + [f"{d}_open" for d in day_prefixes]
            + [f"{d}_close" for d in day_prefixes]
        )
        sched_df = df.select(sched_cols).with_columns(dur_exprs + open_exprs)

        dur_cols = [f"{d}_dur" for d in day_prefixes]
        open_cols = [f"{d}_open_float" for d in day_prefixes]

        days_with_schedule = pl.sum_horizontal(
            [pl.col(c).is_not_null() for c in dur_cols]
        )
        weekly_hours = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in dur_cols])

        night_only = pl.all_horizontal(
            [(pl.col(c).is_null() | (pl.col(c) >= night_start)) for c in open_cols]
        ) & (days_with_schedule > 0)

        step4_df = sched_df.select(
            [
                pk,
                (days_with_schedule == 0).alias("no_schedule"),
                ((days_with_schedule > 0) & (weekly_hours < min_weekly_hours)).alias(
                    "few_hours"
                ),
                night_only.fill_null(False).alias("night_only"),
            ]
        )

        step4_path = temp_dir / "step4_schedules.parquet"
        step4_df.write_parquet(step4_path)

        print("    [Saved] Step 4 written to disk. Freeing memory...")
        del (
            sched_df,
            step4_df,
            dur_exprs,
            open_exprs,
            days_with_schedule,
            weekly_hours,
            night_only,
        )
        gc.collect()

        # ---------------------------------------------------------
        # STEP 5: Consolidation using LazyFrames
        # ---------------------------------------------------------
        print(" -> [Step 5/5] Consolidating intermediate files lazily...")

        # Scan Parquet creates a lazy computational graph (Zero RAM used here)
        lf1 = pl.scan_parquet(step1_path)
        lf2 = pl.scan_parquet(step2_path)
        lf3 = pl.scan_parquet(step3_path)
        lf4 = pl.scan_parquet(step4_path)

        flags_lf = (
            lf1.join(lf2, on=pk, how="inner")
            .join(lf3, on=pk, how="inner")
            .join(lf4, on=pk, how="inner")
        )

        null_issues = [f"null_{col}" for col in key_cols]
        severe_issues = null_issues + [
            "missing_geo",
            "invalid_geo",
            "geo_outlier",
            "no_schedule",
            "useless_name",
            "exact_duplicate",
        ]

        final_flags_lf = flags_lf.with_columns(
            [
                pl.sum_horizontal(severe_issues).alias("severe_issues_count"),
                pl.sum_horizontal(MINOR_ISSUES).alias("minor_issues_count"),
            ]
        ).with_columns(
            [
                (pl.col("severe_issues_count") > 0).alias("has_severe_issue"),
                (pl.col("severe_issues_count") == 0).alias("is_usable"),
            ]
        )

        # Execute the graph and join it back to the original dataframe
        print("    -> Executing joins and returning final dataframe...")
        collected_flags = final_flags_lf.collect()
        final_df = df.join(collected_flags, on=pk, how="left")

        return final_df, severe_issues, MINOR_ISSUES

    finally:
        # --- GUARANTEED MEMORY CLEANUP ---
        print(" -> [Cleanup] Forcing final garbage collection...")
        gc.collect()
        elapsed_time = time.time() - start_time
        print(f" -> Diagnostics pipeline closed after {elapsed_time:.2f} seconds.")
