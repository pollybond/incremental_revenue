import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
import statsmodels.api as sm
from statsmodels.formula.api import ols

from context import GeoPromoContext

logger = logging.getLogger(__name__)

class GeoSolver:
    """
    Солвер для гео-механик.
    Метод: Difference-in-Differences (DiD) с фиксированными эффектами.
    """
    
    def solve(self, ctx: GeoPromoContext) -> Dict[str, Any]:
        try:
            df = ctx.city_revenue_df.copy()
            df["date"] = pd.to_datetime(df["date"])
            
            promo_start = pd.to_datetime(ctx.promo_start)
            promo_end = pd.to_datetime(ctx.promo_end)
            pre_start = pd.to_datetime(ctx.pre_period_start)
            pre_end = pd.to_datetime(ctx.pre_period_end)
            
            # Фильтруем только релевантные периоды
            df = df[(df["date"] >= pre_start) & (df["date"] <= promo_end)]
            
            # Создаем признаки DiD
            df["is_post"] = (df["date"] >= promo_start).astype(int)
            df["is_treated"] = df["city"].apply(
                lambda x: 1 if x in ctx.treated_cities else (0 if x in ctx.control_cities else np.nan)
            )
            df = df.dropna(subset=["is_treated"])
            df["did"] = df["is_treated"] * df["is_post"]
            
            # Проверка данных
            if df["is_treated"].sum() == 0 or df["is_post"].sum() == 0:
                raise ValueError("No treated cities or post-period data found.")
                
            logger.info("Fitting DiD Model (OLS with City & Date FE)...")
            
            # Формула DiD: revenue ~ C(city) + C(date) + did
            # Фиксированные эффекты города и дня убирают сезонность и постоянные различия между городами
            model = ols("revenue ~ C(city) + C(date) + did", data=df).fit(cov_type="HC3")
            
            did_coef = model.params["did"]
            did_se = model.bse["did"]
            did_ci_lower = did_coef - 1.96 * did_se
            did_ci_upper = did_coef + 1.96 * did_se
            
            # Расчет общего инкремента: ATE * количество наблюдений в treatment-периоде
            n_treated_post = len(df[(df["is_treated"] == 1) & (df["is_post"] == 1)])
            total_incremental = did_coef * n_treated_post
            
            return {
                "method": "Difference-in-Differences (DiD)",
                "incremental_revenue": float(total_incremental),
                "confidence_interval": (float(did_ci_lower * n_treated_post), float(did_ci_upper * n_treated_post)),
                "details": {
                    "did_coefficient": float(did_coef),
                    "did_std_error": float(did_se),
                    "n_treated_post_observations": n_treated_post,
                    "model_r_squared": float(model.rsquared),
                    "p_value": float(model.pvalues["did"])
                },
                "status": "SUCCESS"
            }
            
        except Exception as e:
            logger.error(f"Geo Solver failed: {e}", exc_info=True)
            return {
                "method": "Difference-in-Differences (DiD)",
                "incremental_revenue": 0.0,
                "confidence_interval": (0.0, 0.0),
                "status": "FAILED",
                "error": str(e),
                "details": {}
            }