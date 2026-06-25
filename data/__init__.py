"""
Пакет data:
Содержит модули для подключения к БД, загрузки данных,
построения датасетов и обработки признаков.
"""

# 1. Подключение к базе данных
from .db import get_gp_connection

# 2. Загрузчики данных (Loaders)
from .loaders import (
    load_orders_with_geo,
    load_customers,
    load_segments,
    load_revenue_timeseries,
    load_city_revenue_timeseries
)

# 3. Строители датасетов (Builders)
from .builders import (
    build_promo_dataset,
    build_global_dataset,
    build_geo_dataset
)

# 4. Инжиниринг признаков (Features)
from .features import (
    prepare_for_model,
    add_derived_features,
    clip_outliers
)

# Публичный API пакета
__all__ = [
    # Connection
    "get_gp_connection",
    
    # Loaders
    "load_orders_with_geo",
    "load_customers",
    "load_segments",
    "load_revenue_timeseries",
    "load_city_revenue_timeseries",
    
    # Builders
    "build_promo_dataset",
    "build_global_dataset",
    "build_geo_dataset",
    
    # Features
    "prepare_for_model",
    "add_derived_features",
    "clip_outliers"
]