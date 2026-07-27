# geo_utils.py
import json

import polars as pl
from shapely.geometry import Point, shape
from shapely.strtree import STRtree
from tqdm import tqdm


def load_geojson_features(
    geojson_path: str, analyze_state: bool, analyze_municipality: bool
):
    """Carga el GeoJSON y extrae geometrías y metadatos."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        geo_data = json.load(f)

    geometries = []
    states = [] if analyze_state else None
    bottlers = []
    municipalities = [] if analyze_municipality else None

    for feature in geo_data.get("features", []):
        geometries.append(shape(feature["geometry"]))
        props = feature.get("properties", {})

        bottlers.append(props.get("embotelladora", "Unknown"))

        if analyze_state:
            states.append(props.get("estado", "Unknown"))

        if analyze_municipality:
            municipalities.append(props.get("municipio", "Unknown"))

    return geometries, states, bottlers, municipalities


def process_spatial_batches(
    df: pl.DataFrame,
    geometries: list,
    states: list,
    bottlers: list,
    municipalities: list,
    tree: STRtree,
    batch_size: int,
    analyze_state: bool,
    analyze_municipality: bool,
):
    """Executes batched spatial evaluation using the STRtree index."""
    n_rows = df.height

    out_geo_outlier = [False] * n_rows
    out_bottler = [None] * n_rows
    out_state = [None] * n_rows if analyze_state else None
    out_municipality = [None] * n_rows if analyze_municipality else None

    lat_col = df.get_column("latitude").to_numpy()
    lon_col = df.get_column("longitude").to_numpy()
    total_batches = (n_rows + batch_size - 1) // batch_size

    for start_idx in tqdm(
        range(0, n_rows, batch_size),
        total=total_batches,
        desc="[Geo-Spatial] Mapping Points",
        unit="batch",
        ncols=100,
    ):
        end_idx = min(start_idx + batch_size, n_rows)
        lats = lat_col[start_idx:end_idx]
        lons = lon_col[start_idx:end_idx]

        for i in range(len(lats)):
            global_idx = start_idx + i
            lat, lon = lats[i], lons[i]

            if lat is None or lon is None:
                continue

            try:
                pt = Point(float(lon), float(lat))
                indices = tree.query(pt)
                matched = False

                for idx in indices:
                    if geometries[idx].contains(pt):
                        out_bottler[global_idx] = bottlers[idx]
                        if analyze_state:
                            out_state[global_idx] = states[idx]
                        if analyze_municipality:
                            out_municipality[global_idx] = municipalities[idx]

                        out_geo_outlier[global_idx] = False
                        matched = True
                        break

                if not matched:
                    out_bottler[global_idx] = "None"
                    if analyze_state:
                        out_state[global_idx] = "Outlier"
                    if analyze_municipality:
                        out_municipality[global_idx] = "None"
                    out_geo_outlier[global_idx] = True

            except Exception:
                out_bottler[global_idx] = None
                if analyze_state:
                    out_state[global_idx] = None
                if analyze_municipality:
                    out_municipality[global_idx] = None
                out_geo_outlier[global_idx] = True

    return out_geo_outlier, out_state, out_bottler, out_municipality


def print_spatial_summary(
    out_state: list,
    out_bottler: list,
    out_geo_outlier: list,
    out_municipality: list,
    total_rows: int,
    analyze_state: bool,
    analyze_municipality: bool,
):
    """Generates and prints the spatial evaluation summary."""
    temp_dict = {
        "bottler": out_bottler,
        "geo_outlier": out_geo_outlier,
    }

    if analyze_state:
        temp_dict["state"] = out_state
    if analyze_municipality:
        temp_dict["municipality"] = out_municipality

    temp_df = pl.DataFrame(temp_dict)
    outliers_count = temp_df.get_column("geo_outlier").sum()

    # Using only geo_outlier to filter retained points dynamically
    retained_df = temp_df.filter(~pl.col("geo_outlier"))

    print("\n    =============================================")
    print("    📊 SPATIAL EVALUATION SUMMARY")
    print("    =============================================")
    print(f"    Total Evaluated Points: {total_rows:,}")
    print(f"    ❌ Spatial Outliers (Outside area): {outliers_count:,}")
    print(f"    ✅ Retained Points (Inside area): {retained_df.height:,}")

    if retained_df.height > 0:
        group_cols = ["bottler"]
        if analyze_state:
            group_cols.insert(0, "state")
        if analyze_municipality:
            group_cols.append("municipality")

        summary = (
            retained_df.group_by(group_cols)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        print("    ---------------------------------------------")
        print("    Distribution of Retained Points:")
        for row in summary.iter_rows():
            # Build string dynamically based on active toggles
            row_str = "      ->"
            col_idx = 0
            if analyze_state:
                row_str += f" State: {row[col_idx]} |"
                col_idx += 1

            row_str += f" Bottler: {row[col_idx]} |"
            col_idx += 1

            if analyze_municipality:
                row_str += f" Municipality: {row[col_idx]} |"
                col_idx += 1

            row_str += f" Points: {row[col_idx]:,}"
            print(row_str)

    print("    =============================================\n")
