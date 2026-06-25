import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =====================================================
# HELPERS
# =====================================================

def normalize_id(s: pd.Series) -> pd.Series:
    """
    Normalize IDs safely.

    Examples:
    123.0 -> "123"
    " 123 " -> "123"
    """

    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({
            "nan": pd.NA,
            "None": pd.NA,
            "": pd.NA,
        })
    )


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on,
    how: str = "left",
) -> pd.DataFrame:

    if right is None or right.empty:
        return left

    right = right.drop_duplicates(subset=on)

    return left.merge(
        right,
        on=on,
        how=how,
    )


# =====================================================
# CRM BUILDER
# =====================================================

def build_promo_dataset(
    orders: pd.DataFrame,
    customers: pd.DataFrame = None,
    segments: pd.DataFrame = None,
    promo_code: str = None,
    promo_start: str = None,
    promo_end: str = None,
    pre_period_days: int = 30,
    post_period_days: int = 30,
    revenue_col: str = "paid_wo_bonuses_amt",
    datetime_col: str = "created_dttm",
    customer_rk_col: str = "customer_rk",
    customer_id_col: str = "customer_id",
) -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("BUILD PROMO DATASET")
    logger.info("=" * 60)

    # =====================================================
    # DATES
    # =====================================================

    promo_start = pd.to_datetime(promo_start)
    promo_end = pd.to_datetime(promo_end)

    pre_start = promo_start - pd.Timedelta(days=pre_period_days)
    post_end = promo_end + pd.Timedelta(days=post_period_days)

    # =====================================================
    # ORDERS
    # =====================================================

    orders = orders.copy()

    orders[datetime_col] = pd.to_datetime(
        orders[datetime_col],
        errors="coerce",
    )

    orders[customer_rk_col] = normalize_id(
        orders[customer_rk_col]
    )

    orders = orders.dropna(
        subset=[customer_rk_col]
    )

    # =====================================================
    # WINDOW
    # =====================================================

    orders_window = orders[
        (orders[datetime_col] >= pre_start)
        &
        (orders[datetime_col] <= post_end)
    ].copy()

    logger.info(
        f"Orders in window: {len(orders_window):,}"
    )

    logger.info(
        f"Unique customers: "
        f"{orders_window[customer_rk_col].nunique():,}"
    )

    # =====================================================
    # TREATMENT
    # =====================================================

    promo_orders = orders_window[
        (orders_window["promo_code"] == promo_code)
        &
        (orders_window[datetime_col] >= promo_start)
        &
        (orders_window[datetime_col] <= promo_end)
    ]

    treated_rks = (
        promo_orders[customer_rk_col]
        .dropna()
        .unique()
    )

    logger.info(
        f"Treated users: {len(treated_rks):,}"
    )

    # =====================================================
    # REVENUE FEATURES
    # =====================================================

    def agg_revenue(df, start, end):

        return (
            df[
                (df[datetime_col] >= start)
                &
                (df[datetime_col] < end)
            ]
            .groupby(customer_rk_col)[revenue_col]
            .sum()
        )

    revenue_pre = agg_revenue(
        orders_window,
        pre_start,
        promo_start,
    )

    revenue_post = agg_revenue(
        orders_window,
        promo_end,
        post_end,
    )

    # =====================================================
    # BASE DATASET
    # =====================================================
    
    # IMPORTANT:
    # Для causal CRM нельзя брать ВСЕХ пользователей.
    # Иначе control становится несопоставимым с treated.
    #
    # Берем только пользователей,
    # которые были активны ДО промо.
    #
    # Это резко уменьшает selection bias
    # и улучшает overlap.
    
    pre_orders = orders[
        (
            orders[datetime_col] >= pre_start
        )
        &
        (
            orders[datetime_col] < promo_start
        )
    ].copy()
    
    active_pre_rks = (
        pre_orders[customer_rk_col]
        .dropna()
        .unique()
    )
    
    dataset = pd.DataFrame({
        customer_rk_col: active_pre_rks
    })
    
    logger.info(
        f"Users active in pre-period: "
        f"{len(dataset):,}"
    )
    
    # =====================================================
    # TARGETS
    # =====================================================
    
    dataset["revenue_pre"] = (
        dataset[customer_rk_col]
        .map(revenue_pre)
        .fillna(0.0)
    )
    
    dataset["revenue_post"] = (
        dataset[customer_rk_col]
        .map(revenue_post)
        .fillna(0.0)
    )
    
    # target
    dataset["y"] = dataset["revenue_post"]
    
    # treatment flag
    dataset["T"] = (
        dataset[customer_rk_col]
        .isin(treated_rks)
        .astype(int)
    )
    
    # =====================================================
    # MINIMUM ACTIVITY FILTER
    # =====================================================
    
    # Убираем completely cold users.
    #
    # Иначе propensity model
    # идеально разделяет группы.
    
    dataset = dataset[
        dataset["revenue_pre"] > 0
    ].copy()
    
    logger.info(
        f"Users after activity filter: "
        f"{len(dataset):,}"
    )
    
    # =====================================================
    # TREATMENT STATS
    # =====================================================
    
    treated_share = dataset["T"].mean()
    
    logger.info(
        f"Treated share: "
        f"{treated_share:.4f}"
    )
    
    logger.info(
        f"Treated users: "
        f"{dataset['T'].sum():,}"
    )
    
    logger.info(
        f"Control users: "
        f"{(dataset['T'] == 0).sum():,}"
    )
    
    # =====================================================
    # SANITY CHECK
    # =====================================================
    
    if dataset["T"].nunique() < 2:
    
        raise ValueError(
            "Treatment column contains only one class "
            "after pre-period filtering."
        )
    
    logger.info(
        f"Base dataset shape: {dataset.shape}"
    )

    # =====================================================
    # CUSTOMERS BRIDGE
    # =====================================================

    if customers is not None and not customers.empty:

        customers = customers.copy()

        customers[customer_rk_col] = normalize_id(
            customers[customer_rk_col]
        )

        customers[customer_id_col] = normalize_id(
            customers[customer_id_col]
        )

        customers = (
            customers[
                [customer_rk_col, customer_id_col]
            ]
            .dropna(subset=[customer_rk_col])
            .drop_duplicates(subset=[customer_rk_col])
        )

        dataset = safe_merge(
            dataset,
            customers,
            on=customer_rk_col,
        )

        logger.info(
            f"customer_id coverage: "
            f"{dataset[customer_id_col].notna().mean():.2%}"
        )

    # =====================================================
    # SEGMENTS
    # =====================================================

    if (
        segments is not None
        and not segments.empty
        and customer_id_col in dataset.columns
    ):

        segments = segments.copy()

        segments[customer_id_col] = normalize_id(
            segments[customer_id_col]
        )

        segments["report_date"] = pd.to_datetime(
            segments["report_date"],
            errors="coerce",
        )

        segments = segments.dropna(
            subset=[
                customer_id_col,
                "report_date",
            ]
        )

        # latest snapshot
        segments = (
            segments
            .sort_values("report_date")
            .groupby(customer_id_col)
            .tail(1)
        )

        dataset = dataset.merge(
            segments,
            on=customer_id_col,
            how="left",
        )

        logger.info(
            f"segment coverage: "
            f"{dataset['customer_segment'].notna().mean():.2%}"
        )

    # =====================================================
    # HOME CITY
    # =====================================================

    if "city" in orders.columns:

        orders_before = orders[
            orders[datetime_col] < promo_start
        ]

        if not orders_before.empty:

            city_map = (
                orders_before
                .groupby(
                    [customer_rk_col, "city"]
                )
                .size()
                .reset_index(name="cnt")
                .sort_values(
                    ["cnt", "city"],
                    ascending=[False, True],
                )
                .drop_duplicates(
                    subset=[customer_rk_col]
                )
                [
                    [customer_rk_col, "city"]
                ]
                .rename(
                    columns={
                        "city": "home_city"
                    }
                )
            )

            dataset = safe_merge(
                dataset,
                city_map,
                on=customer_rk_col,
            )

    dataset["home_city"] = (
        dataset.get("home_city", "Unknown")
        .fillna("Unknown")
    )

    # =====================================================
    # RFM
    # =====================================================

    orders_before = orders[
        orders[datetime_col] < promo_start
    ]

    if not orders_before.empty:

        rfm = (
            orders_before
            .groupby(customer_rk_col)
            .agg(
                orders_cnt=(
                    "order_id",
                    "count",
                ),
                revenue_sum=(
                    revenue_col,
                    "sum",
                ),
                avg_check=(
                    revenue_col,
                    "mean",
                ),
                discount_share=(
                    "discount_amt",
                    lambda x:
                    (x.fillna(0) > 0).mean(),
                ),
            )
            .reset_index()
        )

        dataset = safe_merge(
            dataset,
            rfm,
            on=customer_rk_col,
        )

    # =====================================================
    # CLEANING
    # =====================================================

    numeric_cols = [
        "revenue_pre",
        "revenue_post",
        "orders_cnt",
        "revenue_sum",
        "avg_check",
        "discount_share",
    ]

    numeric_cols = [
        c for c in numeric_cols
        if c in dataset.columns
    ]

    dataset[numeric_cols] = (
        dataset[numeric_cols]
        .fillna(0.0)
    )

    cat_cols = [
        "customer_segment",
        "age",
        "gender",
        "home_city",
    ]

    cat_cols = [
        c for c in cat_cols
        if c in dataset.columns
    ]

    for col in cat_cols:

        dataset[col] = (
            dataset[col]
            .fillna("Unknown")
            .astype("string")
        )

    logger.info("=" * 60)
    logger.info(
        f"FINAL DATASET SHAPE: {dataset.shape}"
    )
    logger.info("=" * 60)

    return dataset


# =====================================================
# GLOBAL BUILDER
# =====================================================

def build_global_dataset(
    df_ts: pd.DataFrame
) -> pd.DataFrame:

    df = df_ts.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return (
        df
        .set_index("date")
        .sort_index()
        .asfreq("D")
    )


# =====================================================
# GEO BUILDER
# =====================================================

def build_geo_dataset(
    df_geo: pd.DataFrame,
    treated_cities: list,
    control_cities: list,
) -> pd.DataFrame:

    df = df_geo.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["is_treatment"] = np.where(
        df["city"].isin(treated_cities),
        1,
        np.where(
            df["city"].isin(control_cities),
            0,
            np.nan,
        ),
    )

    return df.dropna(
        subset=["is_treatment"]
    )