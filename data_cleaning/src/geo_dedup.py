# src/03_geo_dedup.py
from collections import defaultdict
import numpy as np
import polars as pl
from sklearn.neighbors import BallTree
from unidecode import unidecode


def balltree_spatial_deduplication(
    df: pl.DataFrame,
    radius_meters: int = 50,
    items_to_track: list[str] = None,
) -> tuple[pl.DataFrame, list[str] | None, str | None]:
    """Finds exact duplicates (by normalized name) within a defined spatial radius using a BallTree.

    Keeps a single record per duplicate group and removes the rest. Updates tracked items list upon auditing.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing 'name', 'latitude', 'longitude', and 'dataplor_id' columns.
    radius_meters : int, default 50
        Spatial search radius in meters to evaluate proximity between potential duplicates.
    items_to_track : list[str], optional
        List of specific strings to trace for auditing purposes.

    Returns
    -------
    tuple[pl.DataFrame, list[str] | None, str | None]
        A tuple containing:
        - The deduplicated DataFrame.
        - The updated `items_to_track` list (unmodified if not auditing).
        - An audit report string if `items_to_track` was provided and items were found,
          otherwise None.

    Raises
    ------
    ValueError
        If required columns ('name', 'latitude', 'longitude', 'dataplor_id') are missing.
    """
    required_cols = {"name", "latitude", "longitude", "dataplor_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    is_auditing = items_to_track is not None and len(items_to_track) > 0

    def normalize_name(name):
        if name is None:
            return ""
        return " ".join(unidecode(str(name)).lower().split())

    df_prep = df.with_columns(
        pl.col("name")
        .map_elements(normalize_name, return_dtype=pl.String)
        .alias("name_norm")
    )

    # Separate records with valid coordinates from invalid ones
    valid_mask = (
        pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()
    )
    df_valid = df_prep.filter(valid_mask)

    if df_valid.height == 0:
        cleaned_df = df_prep.drop("name_norm", strict=False)
        return cleaned_df, items_to_track, None

    # Extract arrays to native memory for fast iteration
    coords = np.radians(df_valid.select(["latitude", "longitude"]).to_numpy())
    names = df_valid.get_column("name_norm").to_list()
    ids = df_valid.get_column("dataplor_id").to_list()

    tree = BallTree(coords, metric="haversine")
    earth_radius_m = 6371000.0  # Earth's radius in meters
    indices_list = tree.query_radius(
        coords, r=radius_meters / earth_radius_m
    )

    # Group duplicates by spatial vicinity and name
    groups = defaultdict(set)
    for i, indices in enumerate(indices_list):
        name_i = names[i]
        id_i = ids[i]
        for j in indices:
            if i != j and name_i == names[j]:
                groups[name_i].add(id_i)
                groups[name_i].add(ids[j])

    # Select duplicate IDs to drop (keeping the first encountered ID)
    ids_to_remove = set()
    for name, ids_set in groups.items():
        ids_list = list(ids_set)
        if len(ids_list) > 1:
            ids_to_remove.update(ids_list[1:])

    # Flag dropped rows
    df_clean = df_prep.with_columns(
        pl.when(pl.col("dataplor_id").is_in(list(ids_to_remove)))
        .then(pl.lit(f"Geo_Spatial_Duplicate_(<{radius_meters}m)"))
        .otherwise(pl.lit(None))
        .alias("drop_reason")
    )

    # ==========================================
    # 🚀 FAST MODE (No auditing -> No list modifications)
    # ==========================================
    if not is_auditing:
        df_survivors = df_clean.filter(
            pl.col("drop_reason").is_null()
        ).drop(["name_norm", "drop_reason"])
        return df_survivors, items_to_track, None

    # ==========================================
    # 🔍 AUDIT MODE (Only applies tracking logic here)
    # ==========================================
    df_dropped = df_clean.filter(
        pl.col("drop_reason").is_not_null()
    ).drop("name_norm")
    df_survivors = df_clean.filter(
        pl.col("drop_reason").is_null()
    ).drop(["name_norm", "drop_reason"])

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
        + "\n🎯 AUDIT: MODULE 3 (Spatial Deduplication)\n"
        + "=" * 45
        + "".join(report_lines)
        + "\n"
    )

    return df_survivors, updated_items_to_track, final_report_text