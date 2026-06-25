import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

class DoubleML:
    """
    Double Machine Learning estimator for Average Treatment Effect (ATE).
    Реализует ортогональный момент с кросс-фиттингом (Chernozhukov et al., 2018).
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_col: str = "y",
        treatment_col: str = "T",
        n_splits: int = 5,
        random_state: int = 42,
        lgb_y_params: Optional[dict] = None,
        lgb_t_params: Optional[dict] = None,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.treatment_col = treatment_col
        self.n_splits = n_splits
        self.random_state = random_state

        self.lgb_y_params = lgb_y_params or {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 200,
            "verbosity": -1,
            "random_state": random_state,
        }

        self.lgb_t_params = lgb_t_params or {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 200,
            "verbosity": -1,
            "random_state": random_state,
        }

        self.ate_ = None
        self.se_ = None

    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Fit Double ML estimator via cross-fitting.
        """
        logger.info("Fitting DoubleML (Cross-fitting)...")
        
        X = df[self.feature_cols].values
        Y = df[self.target_col].values
        T = df[self.treatment_col].values
        n = len(df)

        y_res = np.zeros(n)
        t_res = np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]

            # 1. E[Y | X]
            model_y = lgb.LGBMRegressor(**self.lgb_y_params)
            model_y.fit(X_train, Y_train)
            y_hat = model_y.predict(X_test)

            # 2. E[T | X] (Propensity)
            model_t = lgb.LGBMClassifier(**self.lgb_t_params)
            model_t.fit(X_train, T_train)
            t_hat = model_t.predict_proba(X_test)[:, 1]
            
            # 🔒 Clipping для численной стабильности (overlap assumption)
            t_hat = np.clip(t_hat, 0.05, 0.95)

            y_res[test_idx] = Y_test - y_hat
            t_res[test_idx] = T_test - t_hat

        # Final stage: Orthogonal moment condition
        denominator = np.sum(t_res ** 2)
        if denominator == 0:
            raise ValueError("Sum of squared treatment residuals is zero. Check feature leakage or constant T.")
            
        theta = np.sum(t_res * y_res) / denominator

        # Robust standard error via influence function
        psi = t_res * (y_res - theta * t_res)
        var = np.mean(psi ** 2) / (np.mean(t_res ** 2) ** 2)
        se = np.sqrt(var / n)

        self.ate_ = theta
        self.se_ = se

        logger.info(f"🎯 DML ATE: {theta:,.2f} ₽ | SE: {se:,.2f}")

        return {
            "ate": float(theta),
            "se": float(se),
            "ci_lower": float(theta - 1.96 * se),
            "ci_upper": float(theta + 1.96 * se),
        }