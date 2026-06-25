import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# ORDERS
# =========================================================

def load_orders_with_geo(
    conn,
    start_date: str,
    end_date: str,
    promo_code: Optional[str] = None,
    chunksize: int = 200_000,
    limit: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    only_promo: bool = False,
) -> pd.DataFrame:
    """
    Загрузка заказов с географией.

    Parameters
    ----------
    promo_code : str | None
        Код промо.

    only_promo : bool
        Если True:
            загружаются ТОЛЬКО promo-заказы.

        Если False:
            загружаются все заказы
            (включая control).

    limit : int | None
        Ограничение числа строк.

    sample_fraction : float | None
        SQL sampling через random().
    """

    # =====================================================
    # FILTERS
    # =====================================================

    promo_filter = ""

    if only_promo and promo_code:
        promo_filter = """
            AND o.promo_code = %(promo_code)s
        """

    sample_filter = ""

    if sample_fraction:
        sample_filter = f"""
            AND random() < {sample_fraction}
        """

    limit_clause = ""

    if limit:
        limit_clause = f"""
            LIMIT {limit}
        """

    # =====================================================
    # QUERY
    # =====================================================

    query = f"""
        SELECT
            o.order_id,
            o.customer_rk,
            o.created_dttm,
            o.paid_wo_bonuses_amt,
            o.discount_amt,
            o.promo_code,
            ci.city

        FROM grp_bv.c_order o

        JOIN grp_em.dim_store ds
            ON o.store_rk = ds.store_rk
            AND ds.valid_to_dt = '5999-01-01'

        JOIN grp_sandbox.eber_cities ci
            ON ds.city_dk = ci.city_dk

        WHERE o.created_dttm BETWEEN %(start)s AND %(end)s

        {promo_filter}

        {sample_filter}

        {limit_clause}
    """

    # =====================================================
    # PARAMS
    # =====================================================

    params = {
        "start": start_date,
        "end": end_date,
    }

    if only_promo and promo_code:
        params["promo_code"] = promo_code

    # =====================================================
    # LOGGING
    # =====================================================

    logger.info("=" * 60)
    logger.info("LOADING ORDERS WITH GEO")
    logger.info("=" * 60)

    logger.info(f"Period: {start_date} → {end_date}")

    if only_promo:
        logger.info(f"Promo mode enabled: {promo_code}")
    else:
        logger.info("Loading ALL orders (treated + control)")

    if limit:
        logger.warning(
            f"DEBUG LIMIT ENABLED: {limit:,}"
        )

    if sample_fraction:
        logger.warning(
            f"SQL SAMPLE ENABLED: "
            f"{sample_fraction:.2%}"
        )

    # =====================================================
    # LOAD
    # =====================================================

    chunks = []

    total_rows = 0

    for i, chunk in enumerate(
        pd.read_sql(
            query,
            conn,
            params=params,
            chunksize=chunksize,
        )
    ):

        chunk["created_dttm"] = pd.to_datetime(
            chunk["created_dttm"]
        )

        chunks.append(chunk)

        total_rows += len(chunk)

        logger.info(
            f"Chunk #{i + 1}: "
            f"{len(chunk):,} rows "
            f"(total: {total_rows:,})"
        )

    # =====================================================
    # EMPTY CHECK
    # =====================================================

    if not chunks:
        logger.warning("No orders loaded.")
        return pd.DataFrame()

    # =====================================================
    # CONCAT
    # =====================================================

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    logger.info("=" * 60)
    logger.info(
        f"TOTAL ORDERS LOADED: {len(df):,}"
    )
    logger.info("=" * 60)

    return df


# =========================================================
# CUSTOMERS
# =========================================================

def load_customers(
    conn,
    customer_rks=None,
) -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("LOADING CUSTOMERS")
    logger.info("=" * 60)

    where_clause = ""

    params = {}

    if customer_rks is not None:

        customer_rks = [
            str(x).strip()
            for x in customer_rks
            if pd.notna(x)
        ]

        quoted = ",".join(
            f"'{x}'"
            for x in customer_rks
        )

        where_clause = f"""
            WHERE customer_rk IN ({quoted})
        """

    query = f"""
        SELECT
            customer_rk,
            customer_id

        FROM grp_em.dim_customer

        {where_clause}
    """

    df = pd.read_sql(
        query,
        conn,
        params=params,
    )

    df["customer_rk"] = (
        df["customer_rk"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["customer_id"] = (
        df["customer_id"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    logger.info(
        f"Loaded customers: {len(df):,}"
    )

    logger.info(
        f"Unique customer_rk: "
        f"{df['customer_rk'].nunique():,}"
    )

    return df

# =========================================================
# SEGMENTS
# =========================================================

def load_segments(
    conn,
    cutoff_date: str,
    limit: int | None = None,
) -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("LOADING SEGMENTS")
    logger.info("=" * 60)

    limit_clause = (
        f"LIMIT {limit}"
        if limit
        else ""
    )

    query = f"""
        WITH ranked AS (

            SELECT
                customer_id,
                customer_segment,
                age,
                gender,
                report_date,

                ROW_NUMBER() OVER (
                    PARTITION BY customer_id
                    ORDER BY report_date DESC
                ) AS rn

            FROM grp_ss_mrk.ma_customer_segments

            WHERE report_date <= %(cutoff)s
        )

        SELECT
            customer_id,
            customer_segment,
            age,
            gender,
            report_date

        FROM ranked

        WHERE rn = 1

        {limit_clause}
    """

    df = pd.read_sql(
        query,
        conn,
        params={
            "cutoff": cutoff_date,
        },
    )

    df["customer_id"] = (
        df["customer_id"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["report_date"] = pd.to_datetime(
        df["report_date"],
        errors="coerce",
    )

    logger.info(
        f"Loaded rows: {len(df):,}"
    )

    logger.info(
        f"Unique customers: "
        f"{df['customer_id'].nunique():,}"
    )

    return df

# =========================================================
# GLOBAL TS
# =========================================================

def load_revenue_timeseries(
    conn,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    logger.info(
        "LOADING REVENUE TIMESERIES"
    )

    query = """
        SELECT
            created_dttm::date AS date,

            SUM(
                paid_wo_bonuses_amt
            ) AS revenue

        FROM grp_bv.c_order

        WHERE created_dttm
            BETWEEN %(start)s
            AND %(end)s

        GROUP BY 1

        ORDER BY 1
    """

    return pd.read_sql(
        query,
        conn,
        params={
            "start": start_date,
            "end": end_date,
        },
    )


# =========================================================
# GEO TS
# =========================================================

def load_city_revenue_timeseries(
    conn,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    logger.info(
        "LOADING CITY REVENUE TIMESERIES"
    )

    query = """
        SELECT
            created_dttm::date AS date,

            ci.city,

            SUM(
                o.paid_wo_bonuses_amt
            ) AS revenue

        FROM grp_bv.c_order o

        JOIN grp_em.dim_store ds
            ON o.store_rk = ds.store_rk
            AND ds.valid_to_dt = '5999-01-01'

        JOIN grp_sandbox.eber_cities ci
            ON ds.city_dk = ci.city_dk

        WHERE o.created_dttm
            BETWEEN %(start)s
            AND %(end)s

        GROUP BY 1, 2

        ORDER BY 1, 2
    """

    return pd.read_sql(
        query,
        conn,
        params={
            "start": start_date,
            "end": end_date,
        },
    )