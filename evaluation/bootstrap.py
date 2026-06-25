import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple


def bootstrap_incremental_revenue(
    df: pd.DataFrame,
    fit_predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
    treatment_col: str = "T",
    uplift_col: str = "uplift",
    n_bootstrap: int = 5, #200
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Bootstrap confidence interval for incremental revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with treatment indicator
    fit_predict_fn : Callable
        Function that takes a dataframe, fits uplift model
        and returns dataframe with uplift predictions
    treatment_col : str
        Treatment indicator column
    uplift_col : str
        Column with uplift predictions
    n_bootstrap : int
        Number of bootstrap iterations
    ci_level : float
        Confidence level (e.g. 0.95)
    random_state : int

    Returns
    -------
    dict with keys:
        - point_estimate
        - ci_lower
        - ci_upper
        - bootstrap_distribution
    """

    rng = np.random.default_rng(random_state)

    user_idx = df.index.values
    ir_bootstrap = []

    # Point estimate 
    df_point = fit_predict_fn(df)
    point_estimate = df_point.loc[
        df_point[treatment_col] == 1, uplift_col
    ].sum()

    # Bootstrap loop
    for _ in range(n_bootstrap):
        sampled_idx = rng.choice(
            user_idx,
            size=len(user_idx),
            replace=True
        )

        df_sample = df.loc[sampled_idx]

        df_pred = fit_predict_fn(df_sample)

        ir = df_pred.loc[
            df_pred[treatment_col] == 1, uplift_col
        ].sum()

        ir_bootstrap.append(ir)

    ir_bootstrap = np.array(ir_bootstrap)

    alpha = 1 - ci_level
    lower = np.quantile(ir_bootstrap, alpha / 2)
    upper = np.quantile(ir_bootstrap, 1 - alpha / 2)

    return {
        "point_estimate": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_distribution": ir_bootstrap,
    }