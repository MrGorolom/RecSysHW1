# Рекомендации товаров Amazon

Слободин Дмитрий Сергеевич

Домашнее задание 1. Сравнение пяти моделей персональных рекомендаций top-10.

## Датасет

Amazon Reviews 2014, категория Video Games, подвыборка 5-core.
В ней 231 780 отзывов, 24 303 пользователя и 10 672 товара.

- [Первоисточник и описание полей](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
- Отзывы: Video Games → 5-core (231,780 reviews) в первой таблице.
- Метаданные: Per-category files → Video Games → metadata.
- Общий ключ файлов — `asin`; пользователь — `reviewerID`, оценка — `overall`, время — `unixReviewTime`.

Скрипт загружает оба архива в `data/raw/`. Данные исключены из Git.

## Модели

1. Item-based collaborative filtering.
2. Базовый iALS.
3. EASE.
4. Factorization Machine с метаданными товаров.
5. iALS с таргетом на основе оценки и её отклонения от среднего пользователя.

Реализации находятся в `models/`. Использованы NumPy, SciPy и PyTorch.
Отдельно обучен вариант FM только с ID для проверки влияния метаданных.

## Структура

```text
experiments.ipynb       # основной ноутбук
models/                # отдельный модуль для каждой модели
requirements.txt       # зависимости
data_utils.py          # загрузка, очистка и временное разбиение
evaluation.py          # top-10 и метрики
run_experiments.py     # подбор на validation и сравнение на test
results/               # результаты и параметры запусков
RESULTS.md             # отчёт по результатам
tests/                 # проверки формул
```

## Запуск

Python 3.13. Команды для PowerShell, из корня репозитория:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_experiments.py
.\.venv\Scripts\python.exe -m jupyterlab experiments.ipynb
```

В `experiments.ipynb` приведены обработка данных, формулы, параметры и результаты.
Краткий отчёт — [RESULTS.md](RESULTS.md).
Для повторного выполнения ноутбука нужно выбрать ядро созданного окружения
и выполнить все ячейки по порядку.

Архивы загружаются с [сервера McAuley Lab](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/).
Расчёты выполняются на CPU. Для EASE желательно иметь 8 ГБ свободной оперативной памяти.

Результаты сохраняются в `results/` и используются повторно, если код, сетки и данные не менялись.
Повторное обучение:

```powershell
.\.venv\Scripts\python.exe run_experiments.py --force
```

## Оценка

Разбиение по времени: train до 22.08.2013, validation до 18.01.2014, затем test.
Даты на границах относятся к следующей части. Время — UTC.

Релевантные товары имеют оценку 4–5. Параметры выбираются по NDCG@10 на validation.
Затем модели обучаются на train+validation и оцениваются на test.
Дополнительно считаются MAP@10 и Recall@10. Seed — 42.

Все модели ранжируют один train-каталог. Из кандидатов исключаются товары,
которые пользователь уже оценивал. Новые пользователи и товары не входят в основную оценку;
число таких событий указано в отчёте.

Сетки параметров находятся в `run_experiments.py`, версии библиотек и контрольные суммы — в `results/`.

## Источник данных

J. McAuley, C. Targett, J. Shi, A. van den Hengel. *Image-based recommendations on styles and substitutes*. SIGIR, 2015. Ссылки на публикации и правила цитирования приведены на странице датасета.
