import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings

from context import GlobalPromoContext

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class GlobalSolver:
    """
    Солвер для глобальных/общесервисных механик.
    Метод: Causal Impact-like (Time Series Baseline Prediction).
    """
    
    def solve(self, ctx: GlobalPromoContext) -> Dict[str, Any]:
        try:
            df = ctx.revenue_timeseries.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            
            pre_start = pd.to_datetime(ctx.pre_period_start)
            pre_end = pd.to_datetime(ctx.pre_period_end)
            post_start = pd.to_datetime(ctx.post_period_start)
            post_end = pd.to_datetime(ctx.post_period_end)
            
            df_pre = df.loc[pre_start:pre_end]
            df_post = df.loc[post_start:post_end]
            
            if df_pre.empty or df_post.empty:
                raise ValueError("Pre or Post period is empty. Check dates.")
                
            logger.info("Fitting Time Series Baseline on Pre-period...")
            
            # Среднее по дням недели + линейный тренд
            df_pre["day_of_week"] = df_pre.index.dayofweek
            df_pre["trend"] = np.arange(len(df_pre))
            
            # Обучаем baseline на pre-period
            X_pre = df_pre[["day_of_week", "trend"]]
            y_pre = df_pre["revenue"]
            
            # Ridge-регрессия для стабильности
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(X_pre, y_pre)
            
            # Предсказываем post-period
            df_post["day_of_week"] = df_post.index.dayofweek
            df_post["trend"] = np.arange(len(df_pre), len(df_pre) + len(df_post))
            X_post = df_post[["day_of_week", "trend"]]
            
            predicted = model.predict(X_post)
            actual = df_post["revenue"].values
            
            daily_increment = actual - predicted
            total_incremental = np.sum(daily_increment)
            
            # Bootstrap CI для надежности
            logger.info("Computing Bootstrap CI...")
            bootstrap_increments = []
            n_boot = 500
            residuals = daily_increment - np.mean(daily_increment)
            
            for _ in range(n_boot):
                sampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
                boot_inc = np.sum(np.mean(daily_increment) + sampled_residuals)
                bootstrap_increments.append(boot_inc)
                
            ci_lower = np.percentile(bootstrap_increments, 2.5)
            ci_upper = np.percentile(bootstrap_increments, 97.5)
            
            return {
                "method": "Time Series Baseline + Bootstrap",
                "incremental_revenue": float(total_incremental),
                "confidence_interval": (float(ci_lower), float(ci_upper)),
                "details": {
                    "pre_days": len(df_pre),
                    "post_days": len(df_post),
                    "avg_daily_baseline": float(np.mean(predicted)),
                    "avg_daily_actual": float(np.mean(actual))
                },
                "status": "SUCCESS"
            }
            
        except Exception as e:
            logger.error(f"Global Solver failed: {e}", exc_info=True)
            return {
                "method": "Time Series Baseline",
                "incremental_revenue": 0.0,
                "confidence_interval": (0.0, 0.0),
                "status": "FAILED",
                "error": str(e),
                "details": {}
            }