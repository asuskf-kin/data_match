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
    items_to_track: list[str] = None
) -> tuple[pl.DataFrame, list[str] | None, str | None]:
    """
    Applies hours filters, global mislabeling filter, and fuzzy + spatial deduplication.
    Correctly enforces minimum weekly hours (treating missing/null hours as failing the filter)
    and tracks traceability across steps when 'items_to_track' is provided.
    """
    is_auditing = items_to_track is not None and len(items_to_track) > 0

    # Add row index for precise tracking of drop reasons
    df = df.with_row_index("__row_id")

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
    
    # Initialize drop_reason for records failing global keywords
    df = df.with_columns(
        pl.when(pl.col("flag_global_keywords").not_())
        .then(pl.lit("1_Global_Keywords"))
        .otherwise(pl.lit(None))
        .alias("drop_reason")
    )

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
    
    # Corrected logic: Fail if night-only, if hours are missing/null, or if weekly hours < min_hours
    df = df.with_columns(
        (
            (pl.col("_is_night_only") == True) |
            pl.col("_total_weekly_hours").is_null() |
            (pl.col("_total_weekly_hours") < min_hours)
        ).not_().alias("flag_hours")
    )
    
    # Update drop_reason if dropped at hours filter (only if not already dropped)
    df = df.with_columns(
        pl.when(pl.col("drop_reason").is_null() & pl.col("flag_hours").not_())
        .then(pl.lit("2_Hours_Filter"))
        .otherwise(pl.col("drop_reason"))
        .alias("drop_reason")
    )

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

    cands_indices = np.where(cands_mask)[0]
    fuzzy_dropped_row_ids = set()
    if dup_local:
        for idx in cands_indices[list(dup_local)]:
            fuzzy_dropped_row_ids.add(idx)

    flag_fuzzy_list = [row_id not in fuzzy_dropped_row_ids for row_id in df.get_column("__row_id").to_list()]
    df = df.with_columns(pl.Series("flag_fuzzy_dedup", flag_fuzzy_list, dtype=pl.Boolean))
    
    # Update drop_reason if dropped at fuzzy dedup
    df = df.with_columns(
        pl.when(pl.col("drop_reason").is_null() & (pl.col("flag_fuzzy_dedup") == False))
        .then(pl.lit("3_Fuzzy_Dedup"))
        .otherwise(pl.col("drop_reason"))
        .alias("drop_reason")
    )
    
    # Separate survivors and dropped records
    df_dropped = df.filter(pl.col("drop_reason").is_not_null())
    df_survivors = df.filter(pl.col("drop_reason").is_null()).drop(
        ['__row_id', '_name_lower', '_total_weekly_hours', '_is_night_only', 'flag_global_keywords', 'flag_hours', 'flag_fuzzy_dedup', 'drop_reason'], 
        strict=False
    )
    
    if not is_auditing:
        return df_survivors, items_to_track, None

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
            dropped_items_set.add(text)
            reasons = dropped.group_by("drop_reason").agg(
                pl.len().alias("count")
            )
            for row in reasons.iter_rows():
                report_lines.append(
                    f"    ❌ DROPPED: Removed at this step due to: [{row[0]}] ({row[1]} records dropped)."
                )

    updated_items_to_track = [
        item for item in items_to_track if item not in dropped_items_set
    ]

    if len(report_lines) == 0:
        return df_survivors, updated_items_to_track, None

    final_report_text = (
        "=" * 45
        + "\n🎯 AUDIT: MODULE 4 (Hours and Fuzzy Filters)\n"
        + "=" * 45
        + "".join(report_lines)
        + "\n"
    )

    return df_survivors, updated_items_to_track, final_report_text