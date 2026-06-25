import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def check_propensity_overlap(
    df: pd.DataFrame,
    propensity_col: str = "propensity_score",
    treatment_col: str = "T",
) -> Dict[str, Any]:
    """
    Проверка overlap / common support между treated и control.

    Returns
    -------
    dict
        overlap diagnostics
    """

    treated = df.loc[df[treatment_col] == 1, propensity_col]
    control = df.loc[df[treatment_col] == 0, propensity_col]

    if treated.empty or control.empty:
        raise ValueError("Treated or Control group is empty.")

    overlap_min = max(treated.min(), control.min())
    overlap_max = min(treated.max(), control.max())

    overlap_exists = overlap_min < overlap_max

    overlap_share = (
        (
            df[propensity_col]
            .between(overlap_min, overlap_max)
        )
        .mean()
    )

    result = {
        "overlap_exists": overlap_exists,
        "overlap_range": (float(overlap_min), float(overlap_max)),
        "overlap_share": float(overlap_share),
        "treated_range": (
            float(treated.min()),
            float(treated.max())
        ),
        "control_range": (
            float(control.min()),
            float(control.max())
        ),
    }

    logger.info("=== PROPENSITY OVERLAP CHECK ===")
    logger.info(f"Overlap exists: {overlap_exists}")
    logger.info(f"Overlap share: {overlap_share:.3f}")

    if overlap_share < 0.5:
        logger.warning(
            "Low overlap detected. "
            "Causal estimates may be unstable."
        )

    return result


def check_group_balance(
    df: pd.DataFrame,
    treatment_col: str = "T",
    numeric_cols: list | None = None,
) -> pd.DataFrame:
    """
    Сравнение средних признаков между treated/control.

    Parameters
    ----------
    numeric_cols : list
        Numerical columns to compare
    """

    if numeric_cols is None:
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c != treatment_col
        ]

    balance_df = (
        df.groupby(treatment_col)[numeric_cols]
        .mean()
        .T
    )

    balance_df.columns = ["control_mean", "treated_mean"]

    balance_df["abs_diff"] = (
        balance_df["treated_mean"]
        - balance_df["control_mean"]
    ).abs()

    logger.info("=== GROUP BALANCE CHECK ===")
    logger.info(balance_df.head())

    return balance_df


def summarize_treatment(
    df: pd.DataFrame,
    treatment_col: str = "T",
) -> Dict[str, Any]:
    """
    Базовая информация по treatment/control.
    """

    treated_n = int((df[treatment_col] == 1).sum())
    control_n = int((df[treatment_col] == 0).sum())

    total_n = len(df)

    result = {
        "n_total": total_n,
        "n_treated": treated_n,
        "n_control": control_n,
        "treated_share": treated_n / total_n,
    }

    logger.info("=== TREATMENT SUMMARY ===")
    logger.info(result)

    return result