import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Optional

logger = logging.getLogger(__name__)

class XLearner:
    """
    X-Learner for uplift modeling (Kunzel et al., 2019).
    2-ступенчатый алгоритм с импутацией контрфактов и взвешиванием по propensity score.
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_col: str = "y",
        treatment_col: str = "T",
        propensity_col: str = "propensity_score",
        lgb_params: Optional[dict] = None,
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.treatment_col = treatment_col
        self.propensity_col = propensity_col
        self.random_state = random_state

        self.lgb_params = lgb_params or {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.8,
            "metric": "rmse",
            "verbosity": -1,
        }

        # Stage 1: Outcome models
        self.model_mu_0 = None
        self.model_mu_1 = None
        # Stage 2: CATE models
        self.model_tau_0 = None
        self.model_tau_1 = None

    def _get_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Расчет IPW весов: T/e + (1-T)/(1-e)"""
        if self.propensity_col in df.columns:
            e = np.clip(df[self.propensity_col].values, 0.05, 0.95)
            T = df[self.treatment_col].values
            return (T / e) + ((1 - T) / (1 - e))
        return None

    def fit(self, df: pd.DataFrame):
        """
        Stage 1: Train outcome models μ0(x), μ1(x)
        Stage 2: Impute counterfactuals & train CATE models τ0(x), τ1(x)
        """
        logger.info("Fitting X-Learner Stage 1 (Outcome models)...")
        
        X = df[self.feature_cols].values
        Y = df[self.target_col].values
        T = df[self.treatment_col].values

        mask_0 = T == 0
        mask_1 = T == 1

        self.model_mu_0 = lgb.LGBMRegressor(**self.lgb_params, random_state=self.random_state)
        self.model_mu_0.fit(X[mask_0], Y[mask_0])

        self.model_mu_1 = lgb.LGBMRegressor(**self.lgb_params, random_state=self.random_state)
        self.model_mu_1.fit(X[mask_1], Y[mask_1])

        logger.info("Fitting X-Learner Stage 2 (Imputation & CATE models)...")

        # Imputed treatment effects
        D_0 = self.model_mu_1.predict(X[mask_0]) - Y[mask_0] 
        D_1 = Y[mask_1] - self.model_mu_0.predict(X[mask_1])

        weights = self._get_weights(df)
        w_0 = weights[mask_0] if weights is not None else None
        w_1 = weights[mask_1] if weights is not None else None

        self.model_tau_0 = lgb.LGBMRegressor(**self.lgb_params, random_state=self.random_state)
        self.model_tau_0.fit(X[mask_0], D_0, sample_weight=w_0)

        self.model_tau_1 = lgb.LGBMRegressor(**self.lgb_params, random_state=self.random_state)
        self.model_tau_1.fit(X[mask_1], D_1, sample_weight=w_1)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict uplift τ(x) = g(x)*τ0(x) + (1-g(x))*τ1(x)
        """
        X = df[self.feature_cols].values
        
        tau_0 = self.model_tau_0.predict(X)
        tau_1 = self.model_tau_1.predict(X)

        if self.propensity_col in df.columns:
            g = np.clip(df[self.propensity_col].values, 0.05, 0.95)
            uplift = g * tau_0 + (1 - g) * tau_1
        else:
            uplift = 0.5 * tau_0 + 0.5 * tau_1

        df = df.copy()
        df["uplift"] = uplift
        df["tau_0_pred"] = tau_0
        df["tau_1_pred"] = tau_1
        return df

    def uplift_by_segment(self, df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
        """Агрегация uplift по сегментам (только для T=1)"""
        def calc_incremental(group):
            return group.loc[group[self.treatment_col] == 1, "uplift"].sum()

        return (
            df.groupby(segment_col)
            .agg(
                avg_uplift=("uplift", "mean"),
                treated_cnt=(self.treatment_col, "sum"),
                total_cnt=(self.treatment_col, "count"),
                incremental_revenue=("uplift", calc_incremental)
            )
            .reset_index()
            .sort_values("incremental_revenue", ascending=False)
        )