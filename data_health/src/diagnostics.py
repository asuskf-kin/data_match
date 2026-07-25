import json
import re

import polars as pl
from shapely.geometry import Point, shape
from shapely.strtree import STRtree
from unidecode import unidecode


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
    df: pl.DataFrame, geojson_path: str, config: dict = None
) -> list[pl.Series]:
    """Evaluates latitude and longitude against boundaries in GeoJSON using Shapely STRtree in batches.

    Returns a list of Polars Series:
      - 'geo_outlier': Boolean (True if outside all boundaries)
      - 'state': String (State name or None)
      - 'bottler': String (Bottler name or None)
      - 'municipality': String (Municipality name or None)
    """
    print("\n[Geo-Spatial] Starting batched spatial evaluation...")
    print(f"[Geo-Spatial] GeoJSON path: {geojson_path}")
    print(f"[Geo-Spatial] Total rows to evaluate in DataFrame: {df.height:,}")

    # 1. Load GeoJSON
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            geo_data = json.load(f)
        print("[Geo-Spatial] ✅ GeoJSON file loaded successfully.")
    except Exception as e:
        print(f"    [Error] Could not load GeoJSON file at {geojson_path}: {e}")
        return [
            pl.Series("geo_outlier", [False] * df.height, dtype=pl.Boolean),
            pl.Series("state", [None] * df.height, dtype=pl.String),
            pl.Series("bottler", [None] * df.height, dtype=pl.String),
            pl.Series("municipality", [None] * df.height, dtype=pl.String),
        ]

    # 2. Extract geometries and metadata
    geometries = []
    states = []
    bottlers = []
    municipalities = []

    for feature in geo_data.get("features", []):
        geometries.append(shape(feature["geometry"]))
        props = feature.get("properties", {})

        states.append(props.get("estado", "Unknown"))
        bottlers.append(props.get("embotelladora", "Unknown"))
        municipalities.append(props.get("municipio", "Unknown"))

    print(f"[Geo-Spatial] Extracted {len(geometries):,} geometries from GeoJSON.")

    # 3. Create Spatial Index
    tree = STRtree(geometries)
    print("[Geo-Spatial] STRtree spatial index created successfully.")

    # 4. Batched spatial evaluation (Leyendo batch_size de config)
    if config and "parameters" in config and "batch_size" in config["parameters"]:
        batch_size = config["parameters"]["batch_size"]

    print(f"[Geo-Spatial] Processing rows in batches (batch_size: {batch_size:,})...")
    n_rows = df.height

    out_geo_outlier = [False] * n_rows
    out_state = [None] * n_rows
    out_bottler = [None] * n_rows
    out_municipality = [None] * n_rows

    lat_col = df.get_column("latitude").to_numpy()
    lon_col = df.get_column("longitude").to_numpy()
    total_batches = (n_rows + batch_size - 1) // batch_size
    current_batch = 0

    for start_idx in range(0, n_rows, batch_size):
        current_batch += 1
        end_idx = min(start_idx + batch_size, n_rows)
        batch_count = end_idx - start_idx
        print(
            f"[Geo-Spatial] Processing batch {current_batch}/{total_batches} -> rows {start_idx:,} to {end_idx:,} (Batch size: {batch_count:,})..."
        )

        lats = lat_col[start_idx:end_idx]
        lons = lon_col[start_idx:end_idx]

        for i in range(len(lats)):
            global_idx = start_idx + i
            lat = lats[i]
            lon = lons[i]

            # Tu lógica original exacta de validación y control
            if lat is None or lon is None:
                continue

            try:
                pt = Point(float(lon), float(lat))
                indices = tree.query(pt)

                matched = False
                for idx in indices:
                    if geometries[idx].contains(pt):
                        out_state[global_idx] = states[idx]
                        out_bottler[global_idx] = bottlers[idx]
                        out_municipality[global_idx] = municipalities[idx]
                        out_geo_outlier[global_idx] = False
                        matched = True
                        break

                if not matched:
                    out_state[global_idx] = "Outlier"
                    out_bottler[global_idx] = "None"
                    out_municipality[global_idx] = "None"
                    out_geo_outlier[global_idx] = True
            except Exception:
                out_state[global_idx] = None
                out_bottler[global_idx] = None
                out_municipality[global_idx] = None
                out_geo_outlier[global_idx] = True

    print("[Geo-Spatial] Batched spatial mapping completed.")

    # 5. Summary Output
    temp_df = pl.DataFrame(
        {
            "state": out_state,
            "bottler": out_bottler,
            "municipality": out_municipality,
            "geo_outlier": out_geo_outlier,
        }
    )

    outliers_count = temp_df.get_column("geo_outlier").sum()
    retained_df = temp_df.filter(~pl.col("geo_outlier") & pl.col("state").is_not_null())

    print("\n    =============================================")
    print("    📊 SPATIAL EVALUATION SUMMARY")
    print("    =============================================")
    print(f"    Total Evaluated Points: {df.height:,}")
    print(f"    ❌ Spatial Outliers (Outside area): {outliers_count:,}")
    print(f"    ✅ Retained Points (Inside area): {retained_df.height:,}")

    if retained_df.height > 0:
        summary = (
            retained_df.group_by(["state", "bottler", "municipality"])
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        print("    ---------------------------------------------")
        print("    Distribution of Retained Points:")
        for row in summary.iter_rows():
            print(
                f"      -> State: {row[0]} | Bottler: {row[1]} | Municipality: {row[2]} | Points: {row[3]:,}"
            )
    print("    =============================================\n")

    # 6. Return as list of Polars Series
    return [
        pl.Series("geo_outlier", out_geo_outlier, dtype=pl.Boolean),
        pl.Series("state", out_state, dtype=pl.String),
        pl.Series("bottler", out_bottler, dtype=pl.String),
        pl.Series("municipality", out_municipality, dtype=pl.String),
    ]
