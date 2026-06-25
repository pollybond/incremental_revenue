import logging
import pandas as pd
from config.promo_types import PromoMechanicType
from context import CRMPromoContext
from core.factory import IncrementalRevenueFactory
from data.loaders import load_orders_with_geo, load_customers, load_segments
from data.db import get_gp_connection
from config.settings import DB_CONFIG

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RealTest")

def main():
    logger.info("Запуск на реальной акции: ВЕСНАЦВЕТЕТ")

    # Даты загрузки (должны покрывать Pre + Promo + Post)
    # Promo: 26.02.2026 - 26.03.2026
    # Pre (30 дней): с 27.01.2026
    # Post (14 дней): до 09.04.2026
    load_start = "2026-01-20"  # Запас на случай смещений
    load_end   = "2026-04-15"
    
    promo_start = "2026-02-26"
    promo_end   = "2026-03-26"
    promo_code  = "ВЕСНАЦВЕТЕТ"

    try:
        # Загрузка данных из БД
        with get_gp_connection(**DB_CONFIG) as conn:
            logger.info("Загружаем заказы (ВСЕ, включая control-группу)...")
            # ВАЖНО: promo_code=None, чтобы загрузить и тех, кто НЕ брал промо
            orders = load_orders_with_geo(conn, load_start, load_end, promo_code=None)
            logger.info(f"   Загружено заказов: {len(orders):,}")
            if orders.empty:
                raise ValueError("Таблица заказов пуста за указанный период. Проверьте даты и подключение.")

            logger.info("Загружаем справочник клиентов...")
            customers = load_customers(conn)
            logger.info(f"   Загружено клиентов: {len(customers):,}")

            logger.info("Загружаем сегменты (срез на начало акции)...")
            segments = load_segments(conn, cutoff_date=promo_start)
            logger.info(f"   Загружено сегментов: {len(segments):,}")

        # Формируем контекст
        ctx = CRMPromoContext(
            mechanic_type=PromoMechanicType.CRM,
            promo_name="Весна Цветет 2026",
            orders_df=orders,
            customers_df=customers,
            segments_df=segments,
            promo_code=promo_code,
            promo_start=promo_start,
            promo_end=promo_end,
            pre_period_days=30,
            post_period_days=14  
        )

        # Запуск расчёта
        logger.info("Запуск пайплайна: Propensity → X-Learner → DML → Bootstrap...")
        logger.info("Это может занять несколько минут в зависимости от объёма данных.")
        
        result = IncrementalRevenueFactory.calculate(ctx)

        # 4️⃣ Вывод результатов
        print("\n" + "="*60)
        print("РАСЧЁТ ЗАВЕРШЁН")
        print("="*60)
        print(f"Инкрементальная выручка : {result['incremental_revenue']:,.2f} ₽")
        print(f"95% Доверительный интервал: {result['confidence_interval']}")
        print(f"DML ATE (средний эффект)  : {result.get('ate_dml', 'N/A')} ₽")
        print(f"Количество клиентов      : {result.get('details', {}).get('n_customers', 'N/A')}")
        print(f"Доля получивших промо (T): {result.get('details', {}).get('treatment_rate', 0):.1%}")
        print(f" Статус                   : {result['status']}")
        
        if result['status'] == 'FAILED':
            print(f"Ошибка: {result.get('error')}")
        print("="*60 + "\n")

        # Сохраняем предсказания для дальнейшего анализа
        if "df_predictions" in result:
            output_path = "predictions_VESNATSVETET.csv"
            result["df_predictions"].to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info(f"Предсказания сохранены в {output_path}")

    except Exception as e:
        logger.error(f"Критическая ошибка при расчёте: {e}", exc_info=True)

if __name__ == "__main__":
    main()