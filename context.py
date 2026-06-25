from dataclasses import dataclass
import pandas as pd
from config.promo_types import PromoMechanicType

# Базовый контракт (общие поля)
@dataclass(frozen=True)
class BasePromoContext:
    mechanic_type: PromoMechanicType
    promo_name: str

# 1. CRM / Персональные механики
@dataclass(frozen=True)
class CRMPromoContext(BasePromoContext):
    orders_df: pd.DataFrame
    customers_df: pd.DataFrame
    segments_df: pd.DataFrame
    promo_code: str
    promo_start: str
    promo_end: str
    pre_period_days: int = 30
    post_period_days: int = 30

# 2. Глобальные механики (Time Series)
@dataclass(frozen=True)
class GlobalPromoContext(BasePromoContext):
    revenue_timeseries: pd.DataFrame
    pre_period_start: str
    pre_period_end: str
    post_period_start: str
    post_period_end: str
    covariates_df: pd.DataFrame = None

# 3. Гео-механики (Difference-in-Diff)
@dataclass(frozen=True)
class GeoPromoContext(BasePromoContext):
    city_revenue_df: pd.DataFrame
    treated_cities: list
    control_cities: list
    promo_start: str
    promo_end: str
    pre_period_start: str
    pre_period_end: str

# 4. Товарные механики (Item-Level Uplift)
@dataclass(frozen=True)
class ItemPromoContext(BasePromoContext):
    items_df: pd.DataFrame          # Детализация чеков (SKU, order_id, price)
    promo_sku_list: list            # Список товаров, участвующих в акции
    promo_start: str
    promo_end: str
    pre_period_days: int = 30
    post_period_days: int = 14