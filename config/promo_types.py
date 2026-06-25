from enum import Enum
from typing import Dict

class PromoMechanicType(Enum):
    """
    Типы маркетинговых механик для расчета инкремента.
    """
    CRM = "CRM"
    GLOBAL = "GLOBAL"
    GEO = "GEO"
    ITEM = "ITEM"


MECHANIC_DESCRIPTIONS: Dict[PromoMechanicType, str] = {
    PromoMechanicType.CRM: """
    Тип: CRM / Персонализированная механика.
    Суть: Промокод выдан только части пользователей (есть Treatment и Control группы).
    Метод: Double Machine Learning (DML) / X-Learner.

    """,

    PromoMechanicType.GLOBAL: """
    Тип: Глобальная / Общесервисная механика.
    Суть: Акция доступна всем пользователям (нет контрольной группы людей).
    Метод: Causal Impact / Time Series (Сравнение факта с прогнозом).
    
    """,

    PromoMechanicType.GEO: """
    Тип: Гео-механика.
    Суть: Акция работает только в определенных городах/регионах.
    Метод: Difference-in-Differences (DiD).

    """,
    
    PromoMechanicType.ITEM: """
    Тип: Товарная механика (Скидки на конкретные SKU).
    Суть: Скидка применяется к конкретным товарам в корзине.
    Метод: Item-Level Uplift.

    """
}