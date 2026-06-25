import pandas as pd


def trim_by_propensity(
    df: pd.DataFrame,
    lower: float = 0.05,
    upper: float = 0.95,
) -> pd.DataFrame:

    return df[
        (df["propensity_score"] >= lower)
        & (df["propensity_score"] <= upper)
    ].copy()