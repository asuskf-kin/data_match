# src/05_final_regex.py
import polars as pl
import re
from pathlib import Path
from config.keep_words import KEEP_WORDS

def final_keyword_exclusion(
    df: pl.DataFrame, 
    keywords_csv_path: str,
    target_audit: str = None  # <-- PARAMETER FOR FREE TEXT AUDIT
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Filters businesses based on an external keywords CSV file, using complex 
    Regular Expressions while respecting a white-list (KEEP_WORDS).
    If 'target_audit' is provided, it prints the traceability of the matching records.

    Args:
        df (pl.DataFrame): The input Polars DataFrame containing business data (must have a 'name' column).
        keywords_csv_path (str): The file path to the CSV containing the exclusion keywords.
        target_audit (str, optional): A text string to search for in 'name' to trace its execution path.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
            - df_clean (pl.DataFrame): The main dataset with the matched rows removed.
            - df_filtered_out (pl.DataFrame): A dataset of the excluded records, containing 
              the 'dataplor_id' (if present), 'name', and the 'detected_keyword'.
    """
    
    if target_audit:
        print(f"\n{'='*50}\n🔍 AUDITING: '{target_audit}' IN FINAL REGEX EXCLUSION\n{'='*50}")
    
    # 1. Read external keywords
    try:
        # infer_schema_length=0 forces Polars to read the column as a string type
        df_keywords = pl.read_csv(keywords_csv_path, infer_schema_length=0) 
        col_name = df_keywords.columns[0]
        keywords = df_keywords.get_column(col_name).drop_nulls().to_list()
    except Exception as e:
        print(f"Error reading {keywords_csv_path}: {e}. Returning unfiltered dataset.")
        return df, pl.DataFrame()

    if df.is_empty() or not keywords:
        return df, pl.DataFrame()
        
    if target_audit:
        target_rows = df.filter(pl.col("name").str.to_lowercase().str.contains(target_audit.lower()))
        if target_rows.height == 0:
            print("STEP 0: The target record does not exist in the incoming DataFrame.")
        else:
            print(f"STEP 0: Found {target_rows.height} matching record(s) in incoming DataFrame.")

    # 2. Regex transformation helper functions
    def contains_special_characters(kw_list):
        return any(re.search(r'[\[\]().?*+^$|]', kw) for kw in kw_list)

    def remove_accents(text):
        return text.translate(str.maketrans('áéíóúÁÉÍÓÚ', 'aeiouAEIOU'))

    def transform_keywords_to_regex(kw_list):
        tilde_map = {'a':'[aá]','e':'[eé]','i':'[ií]','o':'[oó]','u':'[uú]',
                     'A':'[AÁ]','E':'[EÉ]','I':'[IÍ]','O':'[OÓ]','U':'[UÚ]'}
        trf = []
        for kw in kw_list:
            for letter, pat in tilde_map.items():
                kw = kw.replace(letter, pat)
            # Python 3.7+ escapes spaces as '\ ', we ensure both are covered
            esc = re.escape(kw).replace('\\ ', r'\s*').replace(' ', r'\s*')
            if len(kw) == 4:
                trf.append(r'\b' + esc + r'\b')
            else:
                base = r'\b' + esc + r'\w*'
                trf.append(base)
                if not kw.endswith('s'):
                    trf.append(base + r's?')
        return trf

    def detect_keyword(text, kw_list):
        if not isinstance(text, str):
            return None
        text_lower = text.lower()
        for kw in kw_list:
            if kw.lower() in text_lower:
                return kw
        return None

    kws = keywords + [remove_accents(kw) for kw in keywords if remove_accents(kw) != kw]
    use_regex = contains_special_characters(kws)

    if not use_regex:
        patterns = transform_keywords_to_regex(kws)
    else:
        patterns = [
            r'\b' + re.sub(r'\(([^)]+)\)', r'(?:\1)', kw).replace('\\ ', r'\s*').replace(' ', r'\s*') + r'\w*'
            for kw in kws
        ]

    # Helper function to avoid the Rust regex 10MB memory limit
    def build_chunked_regex_mask(col_name: str, pattern_list: list, chunk_size: int = 150):
        if not pattern_list:
            return pl.lit(False)
            
        exprs = []
        for i in range(0, len(pattern_list), chunk_size):
            chunk = pattern_list[i:i + chunk_size]
            # (?i) enables case-insensitive matching for each chunk
            regex_str = "(?i)" + "|".join(chunk)
            exprs.append(pl.col(col_name).str.contains(regex_str).fill_null(False))
            
        # Returns True if ANY of the chunks match
        return pl.any_horizontal(exprs)

    # 3. Apply Regex and mask exceptions (using chunks)
    # Add a temporary row_id to simulate Pandas' index behavior for later exclusion
    df = df.with_row_index("__row_id")
    
    # Build the masks in safe batches of 150 keywords at a time
    mask_regex = build_chunked_regex_mask("name", patterns, chunk_size=150)
    
    # Also chunk KEEP_WORDS just in case that list grows large in the future
    keep_patterns = [re.escape(w) for w in KEEP_WORDS]
    mask_keep = build_chunked_regex_mask("name", keep_patterns, chunk_size=150)
    
    if target_audit and target_rows.height > 0:
        # Audit how the target responds to the individual masks
        hit_regex = df.filter(pl.col("name").str.to_lowercase().str.contains(target_audit.lower()) & mask_regex)
        hit_keep = df.filter(pl.col("name").str.to_lowercase().str.contains(target_audit.lower()) & mask_keep)
        
        if hit_regex.height > 0:
            print(f"STEP 1: Target MATCHED exclusion regex mask.")
            if hit_keep.height > 0:
                print(f"STEP 2: Target ALSO MATCHED KEEP_WORDS (whitelist mask). Target is PROTECTED.")
            else:
                print(f"STEP 2: Target did NOT match KEEP_WORDS. Target will be REMOVED.")
        else:
            print("STEP 1: Target did NOT match the exclusion regex mask. Target is SAFE.")

    # Filter the subset of data that meets the exclusion regex AND is NOT in the whitelist
    df_temp = df.filter(mask_regex & ~mask_keep)
    
    # Apply the function to recover which keyword triggered the exclusion
    df_temp = df_temp.with_columns(
        pl.col("name")
        .map_elements(lambda x: detect_keyword(x, keywords), return_dtype=pl.String)
        .alias("detected_keyword")
    )

    # 4. Separate excluded records from kept records
    df_filtered_out_full = df_temp.filter(pl.col("detected_keyword").is_not_null())
    
    # Get the simulated IDs to exclude from the main dataframe
    excluded_ids = df_filtered_out_full.get_column("__row_id")
    
    # Maintain the requested layout for the excluded output
    if 'dataplor_id' in df_filtered_out_full.columns:
        df_filtered_out = df_filtered_out_full.select(['dataplor_id', 'name', 'detected_keyword'])
    else:
        df_filtered_out = df_filtered_out_full.select(['name', 'detected_keyword'])
        
    if target_audit and target_rows.height > 0:
        # Find the target in the rejected dataframe
        rejected_target = df_filtered_out.filter(pl.col("name").str.to_lowercase().str.contains(target_audit.lower()))
        if rejected_target.height > 0:
            detected = rejected_target.get_column("detected_keyword").to_list()[0]
            print(f"\n⚠️ FINAL RESULT: Target was REMOVED. The keyword responsible was: '{detected}'")
        else:
            print(f"\n✅ FINAL RESULT: Target SURVIVED and is in the clean dataset.")
        print(f"{'='*50}\n")
    
    # Clean dataframe (everything that is NOT in the excluded IDs), drop the helper __row_id
    df_clean = df.filter(~pl.col("__row_id").is_in(excluded_ids)).drop("__row_id")
    
    return df_clean, df_filtered_out