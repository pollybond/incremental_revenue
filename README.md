# Offline Uplift System
## Causal ML Framework for Incremental Revenue Estimation

Offline Uplift System — framework для оценки инкрементальной выручки
от маркетинговых механик в условиях observational data
(без проведения A/B тестов).

Проект объединяет:

- Causal Inference
- Uplift Modeling
- Double Machine Learning
- Bootstrap Inference

для оценки причинно-следственного эффекта маркетинговых кампаний.
---
# Задача
## Бизнес-проблема

Большинство маркетинговых акций оцениваются через:
- ROI
- post-analysis
- growth vs previous period

Такие подходы не отделяют:
- корреляцию
- сезонность
- selection bias

Система решает задачу оценки:

> "Какую дополнительную выручку действительно вызвала акция?"

## Ключевые возможности

Система автоматически выбирает алгоритм оценки в зависимости от типа механики:

| Тип механики | Сценарий | Метод оценки | Описание |
| :--- | :--- | :--- | :--- |
| **CRM** | Персональные скидки, рассылки | **DML + X-Learner** | Оценка на уровне клиента с устранением селекции (Propensity Score). |
| **GLOBAL** | Общесервисные акции (для всех) | **Causal Impact / Time Series** | Сравнение факта с прогнозом временного ряда (Baseline). |
| **GEO** | Региональные акции (только в МСК) | **Difference-in-Differences (DiD)** | Сравнение динамики выручки в городах-лечения с контрольными городами. |

---

## Источники данных

Проект настроен на работу со схемами Greenplum/PostgreSQL:

* **Транзакции:** `grp_bv.c_order` (заказы), `grp_bv.c_store` (магазины).
* **Клиенты:** `grp_em.dim_customer` (SCD2 справочник).
* **Сегментация:** `grp_ss_mrk.ma_customer_segments` (RFM, возраст, пол).
* **География:** `grp_sandbox.eber_cities` (привязка магазинов к городам).

---
# Архитектура causal pipeline
## CRM Causal Pipeline

```text
Raw Data
    ↓
Dataset Builder
    ↓
Feature Engineering
    ↓
Propensity Modeling
    ↓
Overlap Diagnostics
    ↓
X-Learner Uplift
    ↓
Double ML Validation
    ↓
Bootstrap Confidence Intervals
    ↓
Incremental Revenue
```
---

## Методология

### 1. CRM: Double Machine Learning & X-Learner
Для персонализированных акций, где есть Treatment (получил промо) и Control (не получил) группы.
* **Propensity Scoring:** Оценка вероятности участия в промо $P(T=1|X)$ через LightGBM для устранения смещения отбора.
* **X-Learner:** Двухэтапная модель для оценки гетерогенного эффекта $\tau(x)$.
* **Double ML:** Оценка среднего эффекта (ATE) с использованием кросс-фиттинга и робастных ошибок.

### 2. GLOBAL: Time Series Baseline
Для акций, где Treatment = 100% (например, бесплатная доставка всем).
* Построение базовой линии на исторических данных (Pre-period).
* Расчет инкремента как разницы: $Incremental = Fact_{post} - Predicted_{post}$.

### 3. GEO: Difference-in-Differences (DiD)
Для акций, запущенных в отдельных регионах.
* Использование фиксированных эффектов по городам и времени.
* Сравнение динамики выручки в тестовых городах относительно контрольных.

---

## Структура проекта

## Project Structure

```text
OFFLINE_UPLIFT/
│
├── config/
├── data/
├── estimators/
├── uplift/
├── dml/
├── propensity/
├── evaluation/
├── pipelines/
├── notebooks/
├── reports/
├── artifacts/
│
├── context.py
├── main.py
└── README.md
```
---

## Установка

### 1. Создать виртуальное окружение
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Установить зависимости
```bash
pip install -r requirements.txt
```

### 3. Настройка окружения
Создайте файл `.env` в корне проекта:

```env
# Database (Greenplum/PostgreSQL)
GP_HOST=localhost
GP_PORT=5432
GP_DB=analytics_db
GP_USER=analyst
GP_PASSWORD=secret_password

# App Settings
RANDOM_STATE=42
LOG_LEVEL=INFO
```

---

## Запуск
### Программное использование

```python
from context import CRMPromoContext
from pipelines.crm_pipeline import run_crm_pipeline

ctx = CRMPromoContext(
    promo_name="Summer Sale",
    orders_df=df_orders,
    customers_df=df_customers,
    segments_df=df_segments,
    promo_code="SUMMER24",
    promo_start="2024-07-01",
    promo_end="2024-07-15",
)

result = run_crm_pipeline(ctx)

print(result["incremental_revenue"])
```

## Диагностика причинно-следственной модели

Перед интерпретацией результатов система автоматически выполняет набор проверок качества causal-оценки:

- проверка overlap (пересечения) между Treatment и Control
- сравнение распределений признаков между группами
- оценка доверительных интервалов через Bootstrap
- валидация среднего эффекта через Double Machine Learning (DML)

### Propensity Overlap

Для корректной causal-оценки Treatment и Control группы должны иметь пересечение по распределению propensity score.

Иными словами, для каждого клиента из treatment-группы должны существовать сопоставимые клиенты из control-группы с похожими характеристиками.

При слабом overlap:
- causal estimate становится нестабильным
- возрастает риск selection bias
- uplift может быть переоценен

Система автоматически вычисляет:
- диапазон overlap
- долю наблюдений в common support
- propensity score distribution для обеих групп
---

````md
## Пример результата

```python
{
    "incremental_revenue": 1543000.50,
    "ci_95": (
        1200000.00,
        1800000.00
    ),
    "ate_dml": 50.25,
}
````

### Интерпретация результата

* акция принесла ~1.54 млн ₽ дополнительной выручки
* эффект статистически значим, так как доверительный интервал не пересекает 0
* средний causal uplift составил ~50 ₽ на клиента
* результат подтвержден как uplift-моделью, так и Double ML оценкой

---

## Ограничения

Система работает с observational data и не заменяет полноценные randomized controlled trials (A/B тесты).

Корректность causal-оценки предполагает:

* отсутствие сильных скрытых факторов (unobserved confounders)
* достаточное пересечение Treatment и Control групп
* корректное построение признаков
* отсутствие target leakage
* стабильность поведения клиентов вне воздействия акции

Для GEO-механик дополнительно требуется выполнение предположения parallel trends.

---

## Roadmap

Планируемые улучшения системы:

* Synthetic Control Method
* DR-Learner / T-Learner
* Heterogeneous Treatment Effects
* SHAP-интерпретация uplift-моделей
* Temporal Cross-Validation
* Item-Level Uplift
* Автоматическая генерация causal diagnostics report
* Мониторинг stability drift моделей

---

## Научная ценность проекта

Проект демонстрирует применение современных методов:

* Double Machine Learning
* Uplift Modeling
* Propensity Scoring
* Causal Inference
* Bootstrap Inference

для оценки инкрементальной выручки маркетинговых кампаний
в условиях отсутствия randomized experiments.

Система позволяет перейти от корреляционного анализа маркетинговых акций
к более устойчивой причинно-следственной оценке эффекта.

---

## Автор

**Бондарь Полина Вячеславовна**  