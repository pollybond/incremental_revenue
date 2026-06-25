import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

from context import CRMPromoContext
from data.builders import build_promo_dataset
from data.features import prepare_for_model, add_derived_features
from propensity.propensity_model import PropensityModel
from uplift.predict import predict_incremental_revenue
from evaluation.bootstrap import bootstrap_incremental_revenue
from dml.dml import DoubleML
from config.settings import RANDOM_STATE

logger = logging.getLogger(__name__)

class CRMSolver:
    """
    Солвер для CRM/персонализированных механик.
    Использует Propensity Scoring → X-Learner → DML → Bootstrap CI.
    """
    
    def solve(self, ctx: CRMPromoContext) -> Dict[str, Any]:
        try:
            logger.info("Step 1: Building Uplift Dataset...")
            df = build_promo_dataset(
                orders=ctx.orders_df,
                customers=ctx.customers_df,
                segments=ctx.segments_df,
                promo_code=ctx.promo_code,
                promo_start=ctx.promo_start,
                promo_end=ctx.promo_end,
                pre_period_days=ctx.pre_period_days,
                post_period_days=ctx.post_period_days
            )
            
            logger.info("Step 2: Feature Engineering...")
            df = add_derived_features(df)
            cat_cols = [c for c in ['home_city', 'customer_segment', 'age', 'gender'] if c in df.columns]
            numeric_clip = ['revenue_pre', 'revenue_sum', 'revenue_post']
            df = prepare_for_model(df, cat_cols=cat_cols, numeric_cols_to_clip=numeric_clip)
            
            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            # Формируем список фич. Исключаем служебные колонки и любые нечисловые типы.
            # Это автоматически уберет 'report_dt', который остался строкой.
            exclude_cols = {'y', 'T', 'customer_rk', 'customer_id', 'promo_code', 'report_dt'}
            
            # Разрешаем только числовые и булевы типы данных
            allowed_dtypes = ['int64', 'float64', 'bool', 'int32', 'float32', 'int16', 'float16', 'uint8', 'uint16', 'uint32', 'uint64']
            
            feature_cols = [
                c for c in df.columns 
                if c not in exclude_cols and df[c].dtype in allowed_dtypes
            ]
            
            if not feature_cols:
                raise ValueError("No valid numeric features found after processing!")
                
            logger.info(f"   - Features count: {len(feature_cols)}")
            # -------------------------

            logger.info("Step 3: Propensity Scoring...")
            ps_model = PropensityModel(feature_cols=feature_cols, random_state=RANDOM_STATE)
            df = ps_model.fit_transform(df)
            
            logger.info("Step 4: X-Learner Prediction...")
            uplift_res = predict_incremental_revenue(df, feature_cols)
            total_incremental = uplift_res["incremental_revenue"]
            df_pred = uplift_res["df_predictions"]
            
            logger.info("Step 5: Bootstrap Confidence Intervals...")
            def fit_predict_fn(df_in):
                return predict_incremental_revenue(df_in, feature_cols)["df_predictions"]
                
            # Уменьшил n_bootstrap до 50 для ускорения на локальной машине
            bootstrap_res = bootstrap_incremental_revenue(
                df=df,
                fit_predict_fn=fit_predict_fn,
                n_bootstrap=50 
            )
            
            logger.info("🔹 Step 6: Double ML Validation...")
            dml = DoubleML(feature_cols=feature_cols)
            dml_res = dml.fit(df)
            
            return {
                "method": "DML + X-Learner + Bootstrap",
                "incremental_revenue": float(total_incremental),
                "confidence_interval": (float(bootstrap_res["ci_lower"]), float(bootstrap_res["ci_upper"])),
                "ate_dml": float(dml_res["ate"]),
                "ci_dml": (float(dml_res["ci_lower"]), float(dml_res["ci_upper"])),
                "details": {
                    "n_customers": len(df),
                    "treatment_rate": float(df["T"].mean()),
                    "feature_count": len(feature_cols)
                },
                "status": "SUCCESS",
                "df_predictions": df_pred
            }
            
        except Exception as e:
            logger.error(f"CRM Solver failed: {e}", exc_info=True)
            return {
                "method": "DML + X-Learner",
                "incremental_revenue": 0.0,
                "confidence_interval": (0.0, 0.0),
                "status": "FAILED",
                "error": str(e),
                "details": {}
            }