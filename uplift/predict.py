import logging
import pandas as pd
from .x_learner import XLearner

logger = logging.getLogger(__name__)

def predict_incremental_revenue(
    df: pd.DataFrame,
    feature_cols: list,
) -> dict:
    """
    Fit X-Learner and compute incremental revenue.
    Возвращает суммарный инкремент (₽) и средний на treated-пользователя.
    """
    logger.info("🔹 Fitting X-Learner...")
    model = XLearner(feature_cols=feature_cols)
    model.fit(df)

    df_pred = model.predict(df)

    uplift_col = "uplift"
    if uplift_col not in df_pred.columns:
        available = [c for c in df_pred.columns if "uplift" in c.lower() or "tau" in c.lower() or "cate" in c.lower()]
        if not available:
            raise ValueError(f"❌ XLearner не вернул колонку с апслифтом. Доступные: {df_pred.columns.tolist()}")
        uplift_col = available[0]
        logger.warning(f"⚠️ Используем колонку '{uplift_col}' вместо 'uplift'")

    incremental_revenue_total = float(df_pred.loc[df_pred["T"] == 1, uplift_col].sum())
    n_treated = int(df_pred["T"].sum())
    incremental_revenue_avg = incremental_revenue_total / n_treated if n_treated > 0 else 0.0

    logger.info(f"💰 Total Incremental Revenue: {incremental_revenue_total:,.2f} ₽")
    logger.info(f"📊 Avg Incremental per Treated User: {incremental_revenue_avg:,.2f} ₽")

    return {
        "incremental_revenue": incremental_revenue_total,
        "incremental_revenue_avg": incremental_revenue_avg,
        "df_predictions": df_pred,
    }