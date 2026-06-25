import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def add_cyclical_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Преобразует дату в циклические признаки (sin/cos), чтобы модель понимала сезонность.
    Например: январь близок к декабрю.
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    
    # Месяц (1-12)
    df['month'] = dt.dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # День недели (0-6)
    df['day_of_week'] = dt.dt.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Флаг выходного
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    
    # Чистим исходные колонки, оставляем только полезные
    cols_to_drop = [c for c in ['month', 'day_of_week'] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает производные признаки на основе RFM и Revenue.
    Это помогает модели уловить нелинейные зависимости.
    """
    df = df.copy()
    
    # Логарифм выручки (смягчает выбросы и делает распределение ближе к нормальному)
    if 'revenue_pre' in df.columns:
        df['log_revenue_pre'] = np.log1p(df['revenue_pre'])
    
    # Доля первого заказа (Новый клиент?)
    if 'orders_cnt' in df.columns:
        df['is_new_customer'] = (df['orders_cnt'] == 1).astype(int)
        
    # Интенсивность покупок (Выручка на заказ)
    if 'revenue_sum' in df.columns and 'orders_cnt' in df.columns:
        # Защита от деления на ноль
        df['revenue_per_order'] = df['revenue_sum'] / df['orders_cnt'].replace(0, 1)
        
    return df


def clip_outliers(df: pd.DataFrame, cols: List[str], lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """
    Ограничивает выбросы (Winsorization) для числовых колонок.
    Заменяет значения ниже 1-го перцентиля и выше 99-го перцентиля граничными значениями.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
            
        lower_bound = df[col].quantile(lower_q)
        upper_bound = df[col].quantile(upper_q)
        
        df[col] = df[col].clip(lower_bound, upper_bound)
        
    return df


def prepare_for_model(
    df: pd.DataFrame, 
    cat_cols: List[str], 
    numeric_cols_to_clip: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Главная функция трансформации перед подачей в LightGBM/EconML.
    """
    logger.info("🔧 Applying feature engineering...")
    
    # Обработка выбросов в числах (например, revenue_pre, revenue_sum)
    if numeric_cols_to_clip:
        df = clip_outliers(df, cols=numeric_cols_to_clip)
        logger.info(f"   - Clipped outliers in {numeric_cols_to_clip}")
    
    # Кодирование категорий (One-Hot)
    # drop_first=True убирает мультиколлинеарность (n категорий -> n-1 столбцов)
    if cat_cols:
        existing_cats = [c for c in cat_cols if c in df.columns]
        if existing_cats:
            df = pd.get_dummies(df, columns=existing_cats, drop_first=True, dtype=int)
            logger.info(f"   - Encoded categories: {existing_cats}")
            
    # Приведение типов bool -> int (для совместимости с некоторыми C++ бэкендами)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df