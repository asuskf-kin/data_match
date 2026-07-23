# src/06_crowded_filter.py
import polars as pl
import numpy as np

# A property (shared parent_location) with this many or more *genuinely co-located*
# places is considered too crowded to prospect -> the co-located rows are removed.
CROWDED_PROPERTY_THRESHOLD = 4
# Radius (m) around the group centroid that counts as "the same building/property".
# parent_location has been found to link places very far apart (bad grouping), so a
# group only triggers the crowded rule for the dense sub-cluster within this radius.
SAME_PROPERTY_RADIUS_M = 100.0


def _haversine_m_expr(lat_col: str, lon_col: str, lat_mean_col: str, lon_mean_col: str) -> pl.Expr:
    """
    Polars expression to calculate Haversine distance in meters to a group centroid.
    """
    r = 6_371_000.0
    
    lat = pl.col(lat_col).radians()
    lon = pl.col(lon_col).radians()
    lat0 = pl.col(lat_mean_col).radians()
    lon0 = pl.col(lon_mean_col).radians()
    
    dlat = lat - lat0
    dlon = lon - lon0
    
    a = (dlat / 2).sin() ** 2 + lat.cos() * lat0.cos() * (dlon / 2).sin() ** 2
    a_clipped = pl.when(a > 1.0).then(1.0).when(a < 0.0).then(0.0).otherwise(a)
    
    return 2.0 * r * a_clipped.sqrt().arcsin()


def filter_crowded_same_property(
    dataplor_rawdata: pl.DataFrame,
    bottler_cleaning_folder: str,
    container_name_dest: str,
    logger,
    lat_col: str,
    lon_col: str,
    threshold: int = CROWDED_PROPERTY_THRESHOLD,
    radius_m: float = SAME_PROPERTY_RADIUS_M,
    target_audit: str = None  # Added audit capabilities consistent with your project
) -> pl.DataFrame:
    """
    Removes places co-located on the same property (same parent_location) when there
    are `threshold` or more GENUINELY together.
    """
    column = "parent_location"
    
    # Check if all required columns exist
    missing_cols = [col for col in (column, lat_col, lon_col) if col not in dataplor_rawdata.columns]
    if missing_cols:
        logger.warning(f"Columns {missing_cols} not found; skipping crowded same property filter.")
        return dataplor_rawdata

    if target_audit:
        print(f"\n{'='*50}\n🔍 AUDITING: '{target_audit}' IN CROWDED PROPERTY FILTER\n{'='*50}")

    df = dataplor_rawdata.with_row_index("__row_id")

    # 1. Filter candidates that have valid coordinates and valid parent_location
    valid_cands = df.filter(
        pl.col(column).is_not_null() & 
        pl.col(lat_col).is_not_null() & 
        pl.col(lon_col).is_not_null()
    )

    if valid_cands.is_empty():
        logger.info("Crowded same-property filter: No valid records with coordinates and parent_location.")
        return dataplor_rawdata

    # 2. Compute group centroid & count points per parent_location
    group_stats = valid_cands.group_by(column).agg([
        pl.col(lat_col).mean().alias("_lat_mean"),
        pl.col(lon_col).mean().alias("_lon_mean"),
        pl.len().alias("_group_count")
    ]).filter(pl.col("_group_count") >= threshold)

    if group_stats.is_empty():
        logger.info(f"Crowded same-property filter: no parent_location with >= {threshold} rows.")
        return dataplor_rawdata

    # 3. Join centroid back to candidate points and evaluate Haversine distance
    candidates_with_stats = valid_cands.join(group_stats, on=column, how="inner").with_columns(
        _haversine_m_expr(lat_col, lon_col, "_lat_mean", "_lon_mean").alias("_dist_m")
    )

    # 4. Identify properties that have >= threshold points within radius_m
    dense_clusters = candidates_with_stats.filter(
        pl.col("_dist_m") <= radius_m
    ).group_by(column).agg([
        pl.len().alias("_dense_count"),
        pl.col("__row_id").alias("_ids_to_drop")
    ]).filter(pl.col("_dense_count") >= threshold)

    if dense_clusters.is_empty():
        logger.info(f"Crowded same-property filter: no dense clusters found within {radius_m:.0f}m.")
        return dataplor_rawdata

    # Extract row IDs marked for removal
    drop_ids = dense_clusters.explode("_ids_to_drop").get_column("_ids_to_drop").to_list()
    pruned_props = dense_clusters.height

    # --- AUDIT LOGGING ---
    if target_audit:
        target_rows = df.filter(
            pl.col("name").str.to_lowercase().str.count_matches(target_audit.lower(), literal=True).fill_null(0) > 0
        )
        if not target_rows.is_empty():
            target_dropped = target_rows.filter(pl.col("__row_id").is_in(drop_ids))
            if not target_dropped.is_empty():
                pl_val = target_dropped.get_column(column).to_list()[0]
                print(f"⚠️ TARGET REMOVED: '{target_audit}' belongs to crowded parent_location '{pl_val}' (>= {threshold} points within {radius_m:.0f}m).")
            else:
                print(f"✅ TARGET KEPT: '{target_audit}' survived the crowded property filter.")
        else:
            print(f"STEP 0: Target '{target_audit}' not found in incoming dataset.")
        print(f"{'='*50}\n")

    # 5. Filter out crowded rows
    rows_before = df.height
    df_clean = df.filter(~pl.col("__row_id").is_in(drop_ids)).drop("__row_id")

    logger.info(
        f"Crowded same-property filter (>= {threshold} within {radius_m:.0f}m of centroid): "
        f"{rows_before} -> {df_clean.height}; {pruned_props} crowded properties pruned"
    )

    return df_clean