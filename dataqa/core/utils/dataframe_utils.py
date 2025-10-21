import pandas as pd


def df_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert a dataframe to markdown.
    Output datetime columns in the format of %Y-%m-%d. TODO add support for timestamp.
    """
    if isinstance(df, pd.Series):
        df_copy = df.to_frame()
    else:
        df_copy = df.copy()
    for column in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[column]):
            # Convert datetime columns to the desired string format
            df_copy[column] = df_copy[column].dt.strftime("%Y-%m-%d")

    # Convert the modified DataFrame to Markdown
    markdown_string = df_copy.to_markdown(index=False)
    return markdown_string
