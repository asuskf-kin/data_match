import re

import geopandas as gpd
import polars as pl
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


def get_geo_outlier_series(df: pl.DataFrame, geojson_path) -> pl.Series:
    """
    Identifies spatial outliers in a highly optimized way by extracting unique coordinates.
    """
    print(f"Evaluating spatial outliers using: {geojson_path.name}...")

    # OPTIMIZATION: Extract only unique coordinates to minimize spatial join workload
    # We strictly select only country, latitude, and longitude as requested.
    unique_coords = (
        df.select(["country", "latitude", "longitude"])
        .with_columns(
            [
                pl.col("latitude").cast(pl.Float64, strict=False),
                pl.col("longitude").cast(pl.Float64, strict=False),
            ]
        )
        .drop_nulls(subset=["latitude", "longitude"])
        .unique(subset=["latitude", "longitude"])
    )

    pdf = unique_coords.to_pandas()

    if pdf.empty:
        return pl.Series("geo_outlier", [False] * df.height)

    # Create GeoDataFrame for the unique points
    gdf_points = gpd.GeoDataFrame(
        pdf,
        geometry=gpd.points_from_xy(pdf["longitude"], pdf["latitude"]),
        crs="EPSG:4326",
    )

    # Load boundaries from GeoJSON
    try:
        gdf_poly = gpd.read_file(geojson_path)
        if gdf_poly.crs is None:
            gdf_poly.set_crs("EPSG:4326", inplace=True)
        else:
            gdf_poly = gdf_poly.to_crs("EPSG:4326")
    except Exception as e:
        print(
            f"Warning: Could not load GeoJSON ({e}). Defaulting geo_outlier to False."
        )
        return pl.Series("geo_outlier", [False] * df.height)

    # Perform the spatial join ONLY on unique points
    joined = gpd.sjoin(gdf_points, gdf_poly, how="left", predicate="within")

    # gdf_poly may contain overlapping/nested boundaries (e.g. provinces + cities),
    # so a single point can match multiple polygons and sjoin duplicates that row.
    # Keep only the first match per point so the row count matches gdf_points/pdf.
    joined = joined[~joined.index.duplicated(keep="first")]

    # If the matched polygon index (index_right) is NaN, it fell outside the boundaries
    is_outlier = joined["index_right"].isna()

    # Create a mapping dataframe
    outlier_mapping = pl.DataFrame(
        {
            "latitude": pdf["latitude"].values,
            "longitude": pdf["longitude"].values,
            "geo_outlier": is_outlier.values,
        }
    )

    # Join the evaluated outliers back to the main dataframe
    df_coords = df.select(
        [
            pl.col("latitude").cast(pl.Float64, strict=False),
            pl.col("longitude").cast(pl.Float64, strict=False),
        ]
    )

    mapped_results = df_coords.join(
        outlier_mapping, on=["latitude", "longitude"], how="left"
    )

    return mapped_results.get_column("geo_outlier").fill_null(False)
