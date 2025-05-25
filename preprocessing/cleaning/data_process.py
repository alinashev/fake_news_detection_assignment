import pandas as pd


def process_date(df):
    """
    Parses and cleans the 'date' column in a DataFrame using mixed datetime formats.

    The function:
    - Converts the 'date' column to datetime format using `pd.to_datetime` with mixed formats.
    - Stores the result in a new column 'date_parsed'.
    - Drops rows where the date could not be parsed (NaT).
    - Prints how many rows were removed due to parsing failure.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'date' column of string type.

    Returns:
        pd.DataFrame: A new DataFrame with an added 'date_parsed' column and invalid rows removed.
    """
    original_len = len(df)
    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    df_cleaned = df.dropna(subset=['date_parsed'])
    removed_count = original_len - len(df_cleaned)
    print(f"Removed {removed_count} rows")

    return df_cleaned