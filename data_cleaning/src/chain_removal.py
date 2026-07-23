import re
import polars as pl
from config.patterns_mx import CHAIN_REGEX

def filter_chains_and_duplicates(
    df: pl.DataFrame, 
    filter_duplicates: bool = True, 
    max_appearances: int = 4
) -> pl.DataFrame:
    """
    Filters a DataFrame by removing commercial chains and massive duplicates,
    leveraging the already existing 'name_normalized' column.
    
    Process:
    1. Standardizes quotes/apostrophes directly on 'name_normalized'.
    2. Removes rows where 'name_normalized' matches any pattern in CHAIN_REGEX.
    3. If `filter_duplicates` is True, removes businesses appearing more than `max_appearances` times.
    4. Removes rows where the 'identified_as_chain' column explicitly reads "True".
    
    Args:
        df (pl.DataFrame): The input DataFrame containing a 'name_normalized' column.
        filter_duplicates (bool): Whether to activate the recurrent duplicates filter. Defaults to True.
        max_appearances (int): The maximum allowed occurrences for a single business (if filter is active). Defaults to 4.
        
    Returns:
        pl.DataFrame: A cleaned DataFrame with detected chains removed.
    """
    # Verify the required column exists to avoid cryptic errors later
    if "name_normalized" not in df.columns:
        raise ValueError("Column 'name_normalized' is missing. Run normalize_names first.")

    # Combine the regex list into a single pattern.
    # The (?i) prefix tells Polars' Rust engine to ignore case.
    chain_pattern = "(?i)" + "|".join(f"(?:{p})" for p in CHAIN_REGEX)
    
    # Create a temporary column that forces lowercase and fixes quotes 
    # strictly for filtering purposes.
    df_clean = df.with_columns(
        pl.col("name_normalized")
        .str.replace_all(r"[´`‘’ʼ]", "'")
        .str.to_lowercase()
        .alias("_filter_key")
    )
    
    # 1. Regex filter (commercial chains)
    # Use fill_null(False) to ensure empty fields aren't dropped by mistake.
    df_clean = df_clean.filter(
        ~pl.col("_filter_key").str.contains(chain_pattern).fill_null(False)
    )
    
    # 2. Recurrent duplicates filter (controlled by the user)
    if filter_duplicates:
        df_clean = df_clean.filter(
            pl.col("_filter_key").is_null() | 
            (pl.col("_filter_key").len().over("_filter_key") <= max_appearances)
        )
    
    # 3. Original dataset flag filter (if the column exists)
    if "identified_as_chain" in df_clean.columns:
        df_clean = df_clean.filter(
            pl.col("identified_as_chain").cast(pl.String).fill_null("") != "True"
        )
        
    # Cleanup: drop the temporary filter key, returning the df 
    # with its original 'name_normalized' intact.
    return df_clean.drop("_filter_key")