import logging
from typing import Dict, Any

import pandas as pd

from context import CRMPromoContext

from data.builders import build_promo_dataset
from data.features import (
    prepare_for_model,
    add_derived_features,
)

from propensity.model import PropensityModel

from uplift.predict import (
    predict_incremental_revenue,
)

from evaluation.diagnostics import (
    summarize_treatment,
    check_propensity_overlap,
    check_group_balance,
)

from evaluation.trimming import (
    trim_by_propensity,
)

from config.settings import RANDOM_STATE

logger = logging.getLogger(__name__)


def run_crm_pipeline(
    ctx: CRMPromoContext,
    debug_mode: bool = True,
) -> Dict[str, Any]:

    """
    CRM causal pipeline.
    """

    logger.info("=" * 60)
    logger.info("START CRM PIPELINE")
    logger.info("=" * 60)

    # =====================================================
    # STEP 1: BUILD DATASET
    # =====================================================

    logger.info("STEP 1: BUILD DATASET")

    df = build_promo_dataset(
        orders=ctx.orders_df,
        customers=ctx.customers_df,
        segments=ctx.segments_df,
        promo_code=ctx.promo_code,
        promo_start=ctx.promo_start,
        promo_end=ctx.promo_end,
        pre_period_days=ctx.pre_period_days,
        post_period_days=ctx.post_period_days,
    )

    if df.empty:

        raise ValueError(
            "Dataset is empty after build_promo_dataset()."
        )

    logger.info(
        f"Dataset shape: {df.shape}"
    )

    logger.info(
        f"Dataset memory usage: "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )

    # =====================================================
    # STEP 2: FEATURE ENGINEERING
    # =====================================================

    logger.info(
        "STEP 2: FEATURE ENGINEERING"
    )

    df = add_derived_features(df)

    cat_cols = [
        c for c in [
            "home_city",
            "customer_segment",
            "age",
            "gender",
        ]
        if c in df.columns
    ]

    numeric_clip = [
        c for c in [
            "revenue_pre",
            "revenue_sum",
            "revenue_post",
        ]
        if c in df.columns
    ]

    df = prepare_for_model(
        df,
        cat_cols=cat_cols,
        numeric_cols_to_clip=numeric_clip,
    )

    # =====================================================
    # REMOVE INVALID ROWS
    # =====================================================

    logger.info(
        "STEP 2.1: REMOVE INVALID ROWS"
    )

    df = df.dropna(
        subset=["T", "y"]
    ).copy()

    if df["T"].nunique() < 2:

        raise ValueError(
            "Treatment column contains only one class."
        )

    # =====================================================
    # REMOVE DATETIME COLUMNS
    # =====================================================

    logger.info(
        "STEP 2.2: REMOVE DATETIME COLUMNS"
    )

    datetime_cols = df.select_dtypes(
        include=["datetime64[ns]", "datetime64"]
    ).columns.tolist()

    if datetime_cols:

        logger.warning(
            f"Removing datetime columns: "
            f"{datetime_cols}"
        )

        df = df.drop(
            columns=datetime_cols
        )

    logger.info(
        f"Post-feature shape: {df.shape}"
    )

    # =====================================================
    # STEP 3: FEATURE SELECTION
    # =====================================================

    logger.info(
        "STEP 3: FEATURE SELECTION"
    )

    exclude_cols = {
        "y",
        "T",
        "customer_rk",
        "customer_id",
        "promo_code",
        "propensity_score",
        "uplift",
    }

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
    ]

    if not feature_cols:

        raise ValueError(
            "No valid feature columns found."
        )

    logger.info(
        f"Features count: {len(feature_cols)}"
    )

    # =====================================================
    # STEP 4: TREATMENT SUMMARY
    # =====================================================

    logger.info(
        "STEP 4: TREATMENT SUMMARY"
    )

    treatment_summary = summarize_treatment(df)

    logger.info(
        f"Treated share: "
        f"{treatment_summary['treated_share']:.4f}"
    )

    # =====================================================
    # STEP 5: PROPENSITY MODEL
    # =====================================================

    logger.info(
        "STEP 5: PROPENSITY MODEL"
    )

    ps_model = PropensityModel(
        feature_cols=feature_cols,
        random_state=RANDOM_STATE,
    )

    df = ps_model.fit_transform(df)

    logger.info(
        "Propensity model fitted successfully"
    )

    logger.info(
        f"Propensity min: "
        f"{df['propensity_score'].min():.4f}"
    )

    logger.info(
        f"Propensity max: "
        f"{df['propensity_score'].max():.4f}"
    )

    # =====================================================
    # STEP 6: OVERLAP DIAGNOSTICS
    # =====================================================

    logger.info(
        "STEP 6: OVERLAP DIAGNOSTICS"
    )

    overlap_result = check_propensity_overlap(df)

    logger.info(
        f"Overlap share: "
        f"{overlap_result['overlap_share']:.4f}"
    )

    # =====================================================
    # STEP 6.1: COMMON SUPPORT TRIMMING
    # =====================================================

    logger.info(
        "STEP 6.1: COMMON SUPPORT TRIMMING"
    )

    before_trim = len(df)

    df = trim_by_propensity(
        df,
        lower=0.05,
        upper=0.95,
    )

    after_trim = len(df)

    logger.info(
        f"Rows before trim: {before_trim}"
    )

    logger.info(
        f"Rows after trim: {after_trim}"
    )

    logger.info(
        f"Removed rows: "
        f"{before_trim - after_trim}"
    )

    # =====================================================
    # STEP 6.2: BALANCE CHECK
    # =====================================================

    logger.info(
        "STEP 6.2: BALANCE DIAGNOSTICS"
    )

    balance_numeric_cols = [
        c for c in [
            "revenue_pre",
            "revenue_sum",
            "orders_cnt",
            "avg_check",
        ]
        if c in df.columns
    ]

    balance_df = check_group_balance(
        df,
        numeric_cols=balance_numeric_cols,
    )

    logger.info(
        "Balance diagnostics completed"
    )

    # =====================================================
    # STEP 7: UPLIFT ESTIMATION
    # =====================================================

    logger.info(
        "STEP 7: X-LEARNER UPLIFT"
    )

    uplift_result = predict_incremental_revenue(
        df,
        feature_cols,
    )

    total_incremental = uplift_result[
        "incremental_revenue"
    ]

    df_predictions = uplift_result[
        "df_predictions"
    ]

    logger.info(
        f"Incremental Revenue: "
        f"{total_incremental:,.2f} ₽"
    )

    # =====================================================
    # DEBUG MODE
    # =====================================================

    if debug_mode:

        logger.info("=" * 60)
        logger.info("DEBUG MODE ENABLED")
        logger.info("Bootstrap skipped")
        logger.info("DoubleML skipped")
        logger.info("=" * 60)

        ci_95 = (None, None)

        ate_dml = None

    else:

        # =============================================
        # BOOTSTRAP
        # =============================================

        from evaluation.bootstrap import (
            bootstrap_incremental_revenue
        )

        logger.info(
            "STEP 8: BOOTSTRAP CI"
        )

        def fit_predict_fn(
            df_input: pd.DataFrame
        ) -> pd.DataFrame:

            result = predict_incremental_revenue(
                df_input,
                feature_cols,
            )

            return result[
                "df_predictions"
            ]

        bootstrap_result = (
            bootstrap_incremental_revenue(
                df=df,
                fit_predict_fn=fit_predict_fn,
                n_bootstrap=100,
                random_state=RANDOM_STATE,
            )
        )

        ci_95 = (
            float(
                bootstrap_result["ci_lower"]
            ),
            float(
                bootstrap_result["ci_upper"]
            ),
        )

        logger.info(
            f"95% CI: "
            f"[{ci_95[0]}, {ci_95[1]}]"
        )

        # =============================================
        # DOUBLE ML
        # =============================================

        from dml.dml import DoubleML

        logger.info(
            "STEP 9: DOUBLE ML VALIDATION"
        )

        dml_model = DoubleML(
            feature_cols=feature_cols
        )

        dml_result = dml_model.fit(df)

        ate_dml = float(
            dml_result["ate"]
        )

        logger.info(
            f"DML ATE: {ate_dml}"
        )

    # =====================================================
    # FINISH
    # =====================================================

    logger.info("=" * 60)
    logger.info("CRM PIPELINE FINISHED")
    logger.info("=" * 60)

    return {

        "incremental_revenue": float(
            total_incremental
        ),

        "ci_95": ci_95,

        "ate_dml": ate_dml,

        "treatment_summary": (
            treatment_summary
        ),

        "overlap_result": (
            overlap_result
        ),

        "balance_table": (
            balance_df
        ),

        "df_predictions": (
            df_predictions
        ),

        "feature_importance": (
            ps_model.get_feature_importance()
        ),

        "dataset_shape": df.shape,

        "feature_count": len(
            feature_cols
        ),
    }