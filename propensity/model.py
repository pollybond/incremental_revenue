import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Optional

logger = logging.getLogger(__name__)

class PropensityModel:
    """
    Модель для оценки Propensity Scores (вероятности получения Treatment).
    """

    def __init__(
        self,
        feature_cols: List[str],
        treatment_col: str = "T",
        random_state: int = 42,
        lgb_params: Optional[dict] = None,
    ):
        self.feature_cols = feature_cols
        self.treatment_col = treatment_col
        self.random_state = random_state
        
        self.lgb_params = lgb_params or {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "verbosity": -1,
            "n_estimators": 100,
            "random_state": random_state,
        }
        
        self.model = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обучает модель и возвращает датафрейм с добавленной колонкой propensity_score.
        """
        logger.info("🔹 Fitting Propensity Model...")
        
        X = df[self.feature_cols]
        y = df[self.treatment_col]
        
        # Балансировка классов
        pos_count = (y == 1).sum()
        scale_pos_weight = (y == 0).sum() / pos_count if pos_count > 0 else 1.0
        
        final_params = self.lgb_params.copy()
        final_params["scale_pos_weight"] = scale_pos_weight
        
        self.model = lgb.LGBMClassifier(**final_params)
        self.model.fit(X, y)
        
        # Предсказываем вероятность T=1
        propensity_scores = self.model.predict_proba(X)[:, 1]
        
        df = df.copy()
        df["propensity_score"] = propensity_scores
        
        logger.info(f"   - Mean Propensity Score: {propensity_scores.mean():.4f}")
        logger.info(f"   - Propensity Score Range: [{propensity_scores.min():.4f}, {propensity_scores.max():.4f}]")
        
        if propensity_scores.min() < 0.01 or propensity_scores.max() > 0.99:
            logger.warning("⚠️ Overlap Violation Detected: Extreme propensity scores found.")
            
        return df

    def get_feature_importance(self) -> pd.DataFrame:
        """Возвращает важность признаков propensity модели"""
        if self.model is None:
            return pd.DataFrame()
            
        importances = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
        )