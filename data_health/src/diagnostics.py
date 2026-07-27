import re

import polars as pl
from shapely.strtree import STRtree
from unidecode import unidecode

from src.geo_utils import (
    load_geojson_features,
    print_spatial_summary,
    process_spatial_batches,
)


def get_null_exprs(key_columns):
    """Returns Polars expressions to flag nulls in key columns."""
    return [pl.col(col).is_null().alias(f"null_{col}") for col in key_columns]


def get_geo_exprs():
    """Returns Polars expressions to evaluate coordinate validity."""
    lat = pl.col("latitude").cast(pl.Float64, strict=False)
    lon = pl.col("longitude").cast(pl.Float64, strict=False)

    missing_geo = lat.is_null() | lon.is_null()
    out_of_bounds = (lat.abs() > 90) | (lon.abs() > 180)
    zero_point = (lat == 0) & (lon == 0)
    invalid_geo = (out_of_bounds | zero_point).fill_null(False) & ~missing_geo

    return [missing_geo.alias("missing_geo"), invalid_geo.alias("invalid_geo")]


def get_keyword_expr(keywords):
    """Returns a Polars expression to detect excluded keywords."""
    pat = "(?i)" + "|".join(re.escape(k) for k in keywords)
    clean_name = (
        pl.col("name")
        .fill_null("")
        .map_elements(
            lambda s: unidecode(str(s).lower()).strip(), return_dtype=pl.String
        )
    )
    return clean_name.str.contains(pat).fill_null(False).alias("useless_name")


def get_quality_expr(columns, low_quality_cut):
    """Returns a Polars expression for low quality scores."""
    if "data_quality_confidence_score" in columns:
        q = pl.col("data_quality_confidence_score").cast(pl.Float64, strict=False)
        return (q < low_quality_cut).fill_null(False).alias("low_quality")
    return pl.lit(False).alias("low_quality")


def get_geo_outlier_series(
    df: pl.DataFrame,
    geojson_path: str,
    config: dict = None,
) -> list[pl.Series]:
    """
    Evaluates latitude and longitude against boundaries in GeoJSON.
    """
    params = config.get("parameters", {}) if config else {}
    analyze_state = params.get("analyze_state", True)
    analyze_municipality = params.get("analyze_municipality", False)

    print("\n[Geo-Spatial] Starting batched spatial evaluation...")
    print(f"[Geo-Spatial] GeoJSON path: {geojson_path}")
    print(f"[Geo-Spatial] Total rows to evaluate: {df.height:,}")
    print(f"[Geo-Spatial] State analysis active: {analyze_state}")
    print(f"[Geo-Spatial] Municipality analysis active: {analyze_municipality}")

    try:
        # Call to public function
        geometries, states, bottlers, municipalities = load_geojson_features(
            geojson_path, analyze_state, analyze_municipality
        )
        print(f"[Geo-Spatial] ✅ Extracted {len(geometries):,} geometries.")
    except Exception as e:
        print(f"    [Error] Could not load GeoJSON file at {geojson_path}: {e}")

        # Base error series
        error_series = [
            pl.Series("geo_outlier", [False] * df.height, dtype=pl.Boolean),
            pl.Series("bottler", [None] * df.height, dtype=pl.String),
        ]

        if analyze_state:
            error_series.insert(
                1, pl.Series("state", [None] * df.height, dtype=pl.String)
            )
        if analyze_municipality:
            error_series.append(
                pl.Series("municipality", [None] * df.height, dtype=pl.String)
            )

        return error_series

    tree = STRtree(geometries)
    print("[Geo-Spatial] STRtree spatial index created successfully.")

    batch_size = params.get("batch_size", 100000)
    print(f"[Geo-Spatial] Processing in batches (batch_size: {batch_size:,})...")

    # Call to public function
    out_geo_outlier, out_state, out_bottler, out_municipality = process_spatial_batches(
        df,
        geometries,
        states,
        bottlers,
        municipalities,
        tree,
        batch_size,
        analyze_state,
        analyze_municipality,
    )
    print("[Geo-Spatial] Batched spatial mapping completed.")

    # Call to public function
    print_spatial_summary(
        out_state,
        out_bottler,
        out_geo_outlier,
        out_municipality,
        df.height,
        analyze_state,
        analyze_municipality,
    )

    # Base result series
    result_series = [
        pl.Series("geo_outlier", out_geo_outlier, dtype=pl.Boolean),
        pl.Series("bottler", out_bottler, dtype=pl.String),
    ]

    # Dynamically inject toggled columns
    if analyze_state:
        result_series.insert(1, pl.Series("state", out_state, dtype=pl.String))
    if analyze_municipality:
        result_series.append(
            pl.Series("municipality", out_municipality, dtype=pl.String)
        )

    return result_series
