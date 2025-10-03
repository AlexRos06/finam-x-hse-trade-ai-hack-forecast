# 📦 Подготовка данных для соревнования FORECAST

Этот пакет скриптов предназначен для подготовки данных и оценки решений участников соревнования по прогнозированию финансовых рынков.

## 📋 Содержание

1. **prepare_data.py** — Подготовка и разделение данных на train/test
2. **evaluate_submission.py** — Оценка предсказаний участников
3. **baseline_solution.py** — Baseline решение для примера

---

## 🚀 Быстрый старт

### Шаг 1: Подготовка данных

```bash
python scripts/prepare_data.py
```

**Что делает этот скрипт:**
- ✅ Загружает `data/raw/candles.csv` и `data/raw/news.csv`
- ✅ Вычисляет таргеты (доходности на 1 и 20 дней)
- ✅ Разбивает данные на train/test по времени
- ✅ Создает пакеты для участников (БЕЗ таргетов в test)
- ✅ Создает пакеты для организаторов (С таргетами для проверки)

**Результат:**
```
data/processed/
├── participants/           # ДЛЯ УЧАСТНИКОВ
│   ├── train_candles.csv   # Train с таргетами
│   ├── test_candles.csv    # Test БЕЗ таргетов ⚠️
│   ├── train_news.csv
│   └── test_news.csv
├── organizers/             # ДЛЯ ОРГАНИЗАТОРОВ
│   ├── test_candles_with_targets.csv  # Test С таргетами
│   ├── ground_truth.csv               # Только таргеты
│   └── test_news_full.csv
└── metadata.json
```

### Шаг 2: Создание baseline решения

```bash
python scripts/baseline_solution.py
```

Это создаст файл `baseline_submission.csv` с предсказаниями.

### Шаг 3: Оценка решения

```bash
python scripts/evaluate_submission.py baseline_submission.csv
```

Это выведет метрики и итоговый score.

---

## 📊 Структура данных

### Формат train_candles.csv (для участников)

| Колонка | Описание | Пример |
|---------|----------|--------|
| `open` | Цена открытия | 100.5 |
| `close` | Цена закрытия | 101.2 |
| `high` | Максимальная цена | 102.0 |
| `low` | Минимальная цена | 100.0 |
| `volume` | Объем торгов | 1000000 |
| `begin` | Дата | 2025-01-15 |
| `ticker` | Тикер | SBER |
| `target_return_1d` | **Таргет:** доходность на 1 день | 0.0234 |
| `target_return_20d` | **Таргет:** доходность на 20 дней | 0.1245 |
| `target_direction_1d` | **Таргет:** направление (1=рост, 0=падение) | 1 |
| `target_direction_20d` | **Таргет:** направление (1=рост, 0=падение) | 1 |

### Формат test_candles.csv (для участников)

⚠️ **Внимание:** В test данных НЕТ колонок с таргетами!

| Колонка | Описание |
|---------|----------|
| `open` | Цена открытия |
| `close` | Цена закрытия |
| `high` | Максимальная цена |
| `low` | Минимальная цена |
| `volume` | Объем торгов |
| `begin` | Дата |
| `ticker` | Тикер |

### Формат news.csv

| Колонка | Описание |
|---------|----------|
| `publish_date` | Дата и время публикации |
| `title` | Заголовок новости |
| `publication` | Полный текст |
| `tickers` | Связанные тикеры (может быть пустым) |

---

## 📤 Формат submission файла

Участники должны создать CSV файл со следующей структурой:

```csv
ticker,begin,pred_return_1d,pred_return_20d,pred_prob_up_1d,pred_prob_up_20d
SBER,2025-05-02,0.01,0.05,0.6,0.7
GAZP,2025-05-02,-0.005,0.02,0.45,0.55
...
```

**Обязательные колонки:**
- `ticker` — тикер актива
- `begin` — дата прогноза
- `pred_return_1d` — предсказанная доходность на 1 день (float)
- `pred_return_20d` — предсказанная доходность на 20 дней (float)
- `pred_prob_up_1d` — вероятность роста на 1 день (от 0 до 1)
- `pred_prob_up_20d` — вероятность роста на 20 дней (от 0 до 1)

**Требования:**
- Одна строка на каждую пару (ticker, date) из test данных
- Вероятности должны быть в диапазоне [0, 1]
- Доходности могут быть любыми (но разумными, например [-0.5, 0.5])

---

## 📈 Метрики оценки

### 1. MAE (Mean Absolute Error) для доходностей

Измеряет среднюю абсолютную ошибку в предсказании доходности:

```
MAE = mean(|pred_return - true_return|)
```

**Цель:** Минимизировать (меньше = лучше)

### 2. Accuracy для направления движения

Процент правильно предсказанных направлений (рост/падение):

```
Accuracy = mean(sign(pred_return) == sign(true_return))
```

**Цель:** Максимизировать (больше = лучше)

### 3. Directional Accuracy (альтернативная)

Основана на вероятности:

```
Directional = mean((pred_prob > 0.5) == (true_return > 0))
```

### 4. Итоговый Score (композитный)

```
Score = w1 * (1 - MAE_normalized) + w2 * Accuracy + w3 * DirectionalAccuracy
```

Где:
- `w1 = 0.3` (вес MAE)
- `w2 = 0.4` (вес Accuracy)
- `w3 = 0.3` (вес DirectionalAccuracy)
- Итоговый score усредняется по горизонтам (1д и 20д)

**Диапазон:** [0, 1], где 1 = идеальное решение

---

## 🔧 Настройка параметров

### prepare_data.py

Можно изменить даты разбиения train/test:

```python
preparer = DataPreparer(
    train_end_date="2025-05-01",   # Последняя дата train
    test_start_date="2025-05-02",  # Первая дата test
    test_end_date="2025-06-01"     # Последняя дата test
)
```

### baseline_solution.py

Можно изменить параметры baseline модели:

```python
baseline = BaselineSolution(
    window_size=5  # Размер окна для моментума
)
```

---

## 🎯 Что отдавать участникам vs организаторам

### ✅ Для участников (публичный пакет)

**Файлы для скачивания:**
```
participants_data.zip
├── train_candles.csv      # С таргетами
├── test_candles.csv       # БЕЗ таргетов ⚠️
├── train_news.csv
├── test_news.csv
└── sample_submission.csv  # Пример формата
```

**Инструкции:**
1. Обучайте модель на `train_candles.csv` (таргеты есть)
2. Делайте предсказания для `test_candles.csv` (таргетов нет!)
3. Сохраняйте предсказания в формате `sample_submission.csv`
4. Загружайте submission для оценки

### 🔐 Для организаторов (приватный пакет)

**Файлы НЕ для публикации:**
```
organizers_data.zip
├── test_candles_with_targets.csv  # Полные данные с таргетами
├── ground_truth.csv               # Только таргеты для проверки
└── test_news_full.csv             # Полные новости (с будущими)
```

**Использование:**
```bash
python scripts/evaluate_submission.py participant_submission.csv
```

---

## 💡 Примеры использования

### Пример 1: Полный цикл для организатора

```bash
# 1. Подготовить данные
python scripts/prepare_data.py

# 2. Создать пример baseline
python scripts/baseline_solution.py

# 3. Оценить baseline
python scripts/evaluate_submission.py baseline_submission.csv

# 4. Упаковать данные для участников
cd data/processed/participants
zip -r ../../../participants_data.zip *
```

### Пример 2: Оценка нескольких submission

```bash
for submission in submissions/*.csv; do
    echo "Evaluating $submission"
    python scripts/evaluate_submission.py "$submission" "reports/$(basename $submission .csv)_report.json"
done
```

### Пример 3: Создание dummy submission для тестирования

```bash
python scripts/evaluate_submission.py --create-dummy
```

---

## ⚠️ Важные замечания

### 1. Временная схема (temporal leakage prevention)

- **Цены доступны до момента t:** Можно использовать данные на момент прогноза
- **Новости доступны до t-1:** Задержка в 1 день (реалистичная задержка обработки)
- **Таргет вычисляется на t+N:** Будущие данные не доступны при обучении

### 2. Вычисление таргетов

В baseline используем упрощение: `adj_close = close`

**Формула доходности:**
```
return = (adj_close_{t+N} / adj_close_t) - 1
```

Для production решения рекомендуется:
- Использовать реальные adj_close с учетом сплитов и дивидендов
- Проверить данные на корректность

### 3. Обработка пропущенных значений

- В train: Строки без таргетов удаляются автоматически
- В test: Пропущенные предсказания заполняются нулями (штраф за точность)

### 4. Горизонты прогноза

По умолчанию: 1 день и 20 дней

Можно изменить в `prepare_data.py`:
```python
self.horizons = [1, 20]  # Можно добавить [5, 10, 20] и т.д.
```

---

## 📚 Дополнительные материалы

### Рекомендуемые улучшения baseline

1. **ML модели:**
   - LightGBM / XGBoost для регрессии
   - LSTM / Transformer для временных рядов

2. **Feature engineering:**
   - Технические индикаторы (RSI, MACD, Bollinger Bands)
   - Лаговые признаки (прошлые доходности)
   - Кросс-активные признаки (корреляции между тикерами)

3. **NLP для новостей:**
   - Sentiment analysis (BERT, FinBERT)
   - Topic modeling (LDA, BERTopic)
   - Named Entity Recognition

4. **Ансамбли:**
   - Стекинг моделей
   - Weighted averaging

### Полезные библиотеки

```python
# Time series
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score

# ML models
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# NLP
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Technical indicators
import ta
```

---

## 🤝 Поддержка

При возникновении вопросов или проблем:
1. Проверьте формат данных
2. Убедитесь, что все пути к файлам корректны
3. Проверьте версии библиотек: `pip install -r requirements.txt`

---

## 📝 Лицензия

Этот проект разработан для образовательных целей в рамках хакатона Finam x HSE Trade AI.


