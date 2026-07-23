# src/03_geo_dedup.py
import polars as pl
import numpy as np
from sklearn.neighbors import BallTree
from unidecode import unidecode
from collections import defaultdict

def balltree_spatial_deduplication(df: pl.DataFrame, radius_meters: int = 50) -> pl.DataFrame:
    """
    Finds exact duplicates (by normalized name) within a defined radius using BallTree.
    Keeps a single Dataplor ID per group of duplicates.
    (Based on 2_Deduplicación_geográfica.ipynb)
    """
    
    def normalize_name(name):
        if name is None:
            return ""
        name_str = unidecode(str(name)).lower().strip()
        return name_str.replace("  ", " ")

    df = df.with_columns(
        pl.col("name")
        .map_elements(normalize_name, return_dtype=pl.String)
        .alias("name_norm")
    )
    
    # Separate records with valid coordinates from invalid ones
    valid_mask = pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()
    df_valid = df.filter(valid_mask)

    if df_valid.height == 0:
        return df.drop("name_norm", strict=False)

    # Extract variables to native memory to iterate at O(1) speed
    coords = np.radians(df_valid.select(["latitude", "longitude"]).to_numpy())
    names = df_valid.get_column("name_norm").to_list()
    ids = df_valid.get_column("dataplor_id").to_list()

    tree = BallTree(coords, metric="haversine")
    
    R = 6371000  # Earth's radius in meters
    indices_list = tree.query_radius(coords, r=radius_meters / R)
    
    groups = defaultdict(set)
    
    for i, indices in enumerate(indices_list):
        name_i = names[i]
        id_i = ids[i]
        for j in indices:
            if i != j:
                if name_i == names[j]:
                    groups[name_i].add(id_i)
                    groups[name_i].add(ids[j])
                    
    ids_to_remove = []
    for name, ids_set in groups.items():
        ids_list = list(ids_set)
        if len(ids_list) > 1:
            ids_to_remove.extend(ids_list[1:])
            
    if ids_to_remove:
        df_final = df.filter(~pl.col("dataplor_id").is_in(ids_to_remove))
    else:
        df_final = df
        
    df_final = df_final.drop("name_norm", strict=False)
    
    return df_final