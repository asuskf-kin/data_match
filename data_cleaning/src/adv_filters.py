# src/04_adv_filters.py
import polars as pl
import numpy as np
import re
from sklearn.neighbors import BallTree
from config.global_exclude import GLOBAL_EXCLUDE_KEYWORDS

try:
    from rapidfuzz import fuzz as _rfuzz
    _name_sim = lambda a, b: _rfuzz.token_set_ratio(a, b)
except ImportError:
    from difflib import SequenceMatcher
    _name_sim = lambda a, b: SequenceMatcher(None, a, b).ratio() * 100

def apply_hours_and_fuzzy_filters(
    df: pl.DataFrame, 
    min_hours: int = 20, 
    night_start: int = 19, 
    fuzzy_thresh: int = 80, 
    dist_m: int = 100,
    target_audit: str = None  # <-- PARAMETER FOR FREE TEXT AUDIT
) -> pl.DataFrame:
    """
    Applies hours filters, global mislabeling filter, and fuzzy + spatial deduplication.
    If 'target_audit' is provided, it prints the traceability of the matching records.
    """
    
    if target_audit:
        print(f"\n{'='*50}\n🔍 AUDITING: '{target_audit}'\n{'='*50}")

    # 1. Normalization for name filters
    def normalize_name(name):
        if name is None:
            return ""
        name_str = str(name).lower()
        from unidecode import unidecode
        return unidecode(name_str).strip()

    df = df.with_columns(
        pl.col("name")
        .map_elements(normalize_name, return_dtype=pl.String)
        .alias("_name_lower")
    )
    
    # 2. Global Mislabeling Filter
    pattern_global = '|'.join(re.escape(k) for k in GLOBAL_EXCLUDE_KEYWORDS)
    df = df.with_columns(
        pl.col("_name_lower")
        .str.contains(pattern_global)
        .not_()
        .alias("flag_global_keywords")
    )
    
    if target_audit:
        target_norm = normalize_name(target_audit)
        audit_rows = df.filter(pl.col("_name_lower").str.contains(target_norm))
        if audit_rows.height > 0:
            survived = audit_rows.filter(pl.col("flag_global_keywords") == True).height
            removed = audit_rows.height - survived
            print(f"STEP 2 (Mislabeling): {survived} survived, {removed} removed by global keyword.")
        else:
            print("STEP 2 (Mislabeling): The record does not exist in the original DataFrame.")
    
    # 3. Vectorized / mapped auxiliary hours functions
    DAY_COLS = ['monday_hours','tuesday_hours','wednesday_hours','thursday_hours','friday_hours','saturday_hours','sunday_hours']
    BAD_HOURS = {'Not available', '', 'nan', '00:00-00:00'}

    def parse_hours_duration(h_str):
        if h_str is None or str(h_str).strip() in BAD_HOURS: 
            return None
        try:
            parts = str(h_str).strip().split('-')
            if len(parts) != 2: 
                return None
            def to_float(t):
                h, m = t.strip().split(':')
                return int(h) + int(m) / 60
            o, c = to_float(parts[0]), to_float(parts[1])
            duration = (24 - o) + c if c < o else c - o
            return duration if duration > 0 else None
        except: 
            return None

    def opening_hour(h_str):
        if h_str is None or str(h_str).strip() in BAD_HOURS: 
            return None
        try:
            parts = str(h_str).strip().split('-')
            if len(parts) != 2: 
                return None
            h, m = parts[0].strip().split(':')
            return int(h) + int(m) / 60
        except: 
            return None

    def compute_hours_row(row_dict):
        durations = [parse_hours_duration(row_dict.get(c)) for c in DAY_COLS]
        valid_durations = [d for d in durations if d is not None]
        total_weekly = sum(valid_durations) if valid_durations else None
        
        opens = [opening_hour(row_dict.get(c)) for c in DAY_COLS]
        valid_opens = [o for o in opens if o is not None]
        
        is_night = None
        if valid_opens:
            is_night = all(o >= night_start for o in valid_opens)
            
        return {"_total_weekly_hours": total_weekly, "_is_night_only": is_night}

    hours_computed = df.select(DAY_COLS).to_dicts()
    computed_results = [compute_hours_row(row) for row in hours_computed]
    
    total_weekly_list = [res["_total_weekly_hours"] for res in computed_results]
    is_night_list = [res["_is_night_only"] for res in computed_results]

    df = df.with_columns([
        pl.Series("_total_weekly_hours", total_weekly_list, dtype=pl.Float64),
        pl.Series("_is_night_only", is_night_list, dtype=pl.Boolean)
    ])
    
    df = df.with_columns(
        (
            (pl.col("_is_night_only") == True) |
            (pl.col("_total_weekly_hours").is_not_null() & (pl.col("_total_weekly_hours") < min_hours))
        ).not_().alias("flag_hours")
    )
    
    if target_audit and audit_rows.height > 0:
        audit_rows = df.filter(pl.col("_name_lower").str.contains(target_norm))
        survived = audit_rows.filter(pl.col("flag_hours") == True).height
        removed = audit_rows.height - survived
        print(f"STEP 3 (Hours): {survived} survived, {removed} removed (night only or < {min_hours}h).")

    # 4. Fuzzy + Spatial Filter
    cands_mask_expr = (
        pl.col("flag_global_keywords") & 
        pl.col("flag_hours") & 
        pl.col("latitude").is_not_null() & 
        pl.col("longitude").is_not_null()
    )
    
    cands_mask = df.select(cands_mask_expr).to_series().to_numpy()
    df_cands = df.filter(cands_mask_expr)
    
    dup_local = set()
    if df_cands.height > 0:
        coords_rad = np.radians(df_cands.select(["latitude", "longitude"]).to_numpy())
        tree = BallTree(coords_rad, metric='haversine')
        nbrs_list = tree.query_radius(coords_rad, r=dist_m / 6_371_000)
        
        cands_names = df_cands.get_column("_name_lower").to_list()
        
        for i, nbrs in enumerate(nbrs_list):
            if i in dup_local: 
                continue
            name_i = cands_names[i]
            for j in nbrs:
                if j <= i or j in dup_local: 
                    continue
                sim_score = _name_sim(name_i, cands_names[j])
                if sim_score >= fuzzy_thresh:
                    dup_local.add(j)
                    
                    # Audit log if it collides with your target
                    if target_audit:
                        if target_norm in name_i or target_norm in cands_names[j]:
                            print(f"\n⚠️ STEP 4 (Deduplication): Collision detected!")
                            print(f"   > Kept: '{name_i}'")
                            print(f"   > Removed: '{cands_names[j]}'")
                            print(f"   > Similarity: {sim_score:.1f}%")

    cands_indices = np.where(cands_mask)[0]
    dup_df_idx = set(cands_indices[list(dup_local)]) if dup_local else set()
    
    flag_fuzzy_list = [i not in dup_df_idx for i in range(df.height)]
    df = df.with_columns(pl.Series("flag_fuzzy_dedup", flag_fuzzy_list, dtype=pl.Boolean))
    
    # 5. Filter the dataframe by combining the flags
    df_clean = df.filter(
        pl.col("flag_global_keywords") & 
        pl.col("flag_hours") & 
        pl.col("flag_fuzzy_dedup")
    )
    
    if target_audit:
        final_count = df_clean.filter(pl.col("_name_lower").str.contains(target_norm)).height
        print(f"\n🏁 FINAL RESULT: {final_count} record(s) of '{target_audit}' reached the final DataFrame.")
        print(f"{'='*50}\n")
    
    # Clean up temporary columns
    cols_to_drop = ['_name_lower', '_total_weekly_hours', '_is_night_only', 'flag_global_keywords', 'flag_hours', 'flag_fuzzy_dedup']
    df_clean = df_clean.drop(cols_to_drop, strict=False)
    return df_clean