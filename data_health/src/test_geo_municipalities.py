import json

import polars as pl
from shapely.geometry import Point, shape
from shapely.strtree import STRtree
from tqdm import tqdm


def evaluate_points_in_area(
    df: pl.DataFrame, geojson_path: str, config: dict = None
) -> pl.DataFrame:
    """Filters points that fall inside the GeoJSON area and assigns

    their state, bottler, and municipality using batches.

    Returns a DataFrame containing ONLY the valid points (survivors).
    """
    print(f"Starting spatial evaluation with file: {geojson_path}")

    # 1. Load the GeoJSON file
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            geo_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading GeoJSON: {e}")
        return df

    # 2. Extract geometries and properties
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

    # 3. Create Spatial Index for ultra-fast searches
    tree = STRtree(geometries)

    # 4. Get batch_size from config (default to 50000 if not provided)
    batch_size = 50000
    if config and "parameters" in config and "batch_size" in config["parameters"]:
        batch_size = config["parameters"]["batch_size"]

    n_rows = df.height
    print(f"Evaluating {n_rows:,} coordinates in batches of {batch_size:,}...")

    out_state = [None] * n_rows
    out_bottler = [None] * n_rows
    out_municipality = [None] * n_rows
    out_inside = [False] * n_rows

    lat_col = df.get_column("latitude").to_numpy()
    lon_col = df.get_column("longitude").to_numpy()

    total_batches = (n_rows + batch_size - 1) // batch_size

    for current_batch, start_idx in enumerate(
        tqdm(
            range(0, n_rows, batch_size),
            total=total_batches,
            desc="[Geo-Spatial] Processing Batches",
        ),
        1,
    ):
        end_idx = min(start_idx + batch_size, n_rows)
        lats = lat_col[start_idx:end_idx]
        lons = lon_col[start_idx:end_idx]

        for i in range(len(lats)):
            global_idx = start_idx + i
            lat = lats[i]
            lon = lons[i]

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
                        out_inside[global_idx] = True
                        matched = True
                        break

                if not matched:
                    out_inside[global_idx] = False
            except Exception:
                out_inside[global_idx] = False

    # 5. Build evaluated DataFrame
    df_evaluated = df.with_columns(
        [
            pl.Series("state", out_state),
            pl.Series("bottler", out_bottler),
            pl.Series("municipality", out_municipality),
            pl.Series("inside_area", out_inside, dtype=pl.Boolean),
        ]
    )

    # 6. Filter to keep ONLY the points inside the area
    df_valid_points = df_evaluated.filter(pl.col("inside_area") == True)
    points_outside = df.height - df_valid_points.height

    # 7. Print quick summary report
    print("\n" + "=" * 45)
    print("📊 SPATIAL EVALUATION RESULTS")
    print("=" * 45)
    print(f"Original points: {df.height:,}")
    print(f"❌ Points outside area (removed): {points_outside:,}")
    print(f"✅ Points inside area (retained): {df_valid_points.height:,}")

    if df_valid_points.height > 0:
        summary = (
            df_valid_points.group_by(["state", "bottler", "municipality"])
            .agg(pl.len().alias("total_points"))
            .sort("total_points", descending=True)
        )
        print("\nDistribution of retained points:")
        for row in summary.iter_rows():
            print(
                f"  -> State: {row[0]} | Bottler: {row[1]} | Municipality: {row[2]} | Points: {row[3]:,}"
            )
    print("=" * 45 + "\n")

    return df_valid_points
