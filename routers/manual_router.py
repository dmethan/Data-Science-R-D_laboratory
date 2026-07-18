# === Standard library ===

import base64
import datetime
import io
import json
import os
import pickle
import zipfile

# === Third-party libraries ===

import matplotlib
matplotlib.use("Agg")  # використовуємо бекенд без GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import scipy.stats as stats

# FastAPI

from fastapi import APIRouter, Request, UploadFile, File, Body, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

# === Sklearn: model selection ===

from sklearn.model_selection import train_test_split, RandomizedSearchCV

# === Sklearn: preprocessing ===

from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

# === Sklearn: linear models ===

from sklearn.linear_model import LogisticRegression, LinearRegression

# === Sklearn: ensemble models ===

from sklearn.ensemble import (
RandomForestClassifier,
RandomForestRegressor,
GradientBoostingClassifier,
GradientBoostingRegressor
)

# === Sklearn: tree & SVM ===

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# === Sklearn: clustering ===

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture

# === Sklearn: multioutput ===

from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

# === Sklearn: metrics ===

from sklearn.metrics import (
    # === КЛАСИФІКАЦІЯ ===
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,

    # === МУЛЬТИКЛАСИФІКАЦІЯ ===
    # (ті ж самі метрики, але застосовуються з average='macro'/'weighted')
    precision_recall_curve,

    # === РЕГРЕСІЯ ===
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    mean_squared_log_error,
    explained_variance_score,

    # === КЛАСТЕРИЗАЦІЯ ===
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score, silhouette_samples
)


# === Local modules ===

from logging_config import logger

#  Імпортуємо роутер для "Professional mode"
router = APIRouter(prefix="/manual_mode", tags=["Professional mode"])

#  Підключаємо шаблони Jinja2
templates = Jinja2Templates(directory="templates")

#  Глобальний стан для збереження даних та історії трансформацій
basic_state = {
    "dataframe": None,
    "model": None,
    "task_type": None,
    "target_column": None,
    "feature_columns": None,
    "transformations": []
}

#  Глобальний словник для збереження інформації про датасет
dataset_info = {}

# ============================================================
# 🔹 Головна сторінка Professional Mode
# ============================================================
@router.get("/", response_class=HTMLResponse)
async def professional_home_page(request: Request):
    logger.info(">>> Відображення головної сторінки Professional Mode")
    return templates.TemplateResponse(
        "professional/manual_mode_index.html",
        {"request": request}
    )

# ============================================================
# 🔹 Логування трансформацій датасету
# ============================================================
def log_transformation(action, column=None, method=None):
    """
    Додає запис про виконану трансформацію у глобальний стан.
    :param action: Назва дії (наприклад, 'Заповнення пропусків')
    :param column: Колонка, до якої застосовано дію
    :param method: Метод трансформації (наприклад, 'mean', 'one_hot')
    """
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "column": column,
        "method": method
    }
    basic_state["transformations"].append(entry)
    logger.info(f">>> Додано трансформацію: {action}, колонка={column}, метод={method}")

# ============================================================
#  Отримання історії трансформацій
# ============================================================
@router.get("/transformations")
async def get_transformations():
    """
    Повертає історію виконаних трансформацій у форматі JSON.
    """
    logger.info(">>> Запит історії трансформацій")
    history = basic_state.get("transformations", [])
    logger.debug(f">>> Кількість трансформацій у історії: {len(history)}")
    return {"history": history}


# ============================================================
# 🔹 Збереження історії трансформацій у файл .log
# ============================================================
@router.get("/save_transformations_log")
async def save_transformations_log():
    """
    Зберігає всі виконані трансформації у текстовий файл transformations.log.
    """
    filename = "transformations.log"
    logger.info(">>> Етап: Збереження історії трансформацій у transformations.log")

    try:
        with open(filename, "w", encoding="utf-8") as f:
            for h in basic_state["transformations"]:
                f.write(f"{h['timestamp']} | {h['action']} | {h['column']} | {h['method']}\n")

        # Логування кількості трансформацій
        logger.info(f">>> Успішно збережено {len(basic_state['transformations'])} трансформацій")
        if basic_state["transformations"]:

            # Логування прикладу першої трансформації
            first = basic_state["transformations"][0]
            logger.debug(f"Приклад першої трансформації: {first}")
        else:
            logger.debug("Історія трансформацій порожня")

    except Exception as e:
        logger.error(f"Помилка при збереженні трансформацій: {e}")
        raise

    return FileResponse(filename, media_type="text/plain", filename=filename)


# ============================================================
# 🔹 Експорт метаданих трансформацій у JSON
# ============================================================
@router.get("/export_metadata_json")
async def export_metadata_json():
    """
    Експортує всі трансформації у форматі JSON (metadata.json).
    """
    filename = "metadata.json"
    logger.info(">>> Етап: Експорт метаданих трансформацій у metadata.json")

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(basic_state["transformations"], f, ensure_ascii=False, indent=2)

        # Логування кількості трансформацій
        logger.info(f">>> Успішно експортовано {len(basic_state['transformations'])} трансформацій у JSON")
        if basic_state["transformations"]:
            # Логування прикладу першої трансформації
            first = basic_state["transformations"][0]
            logger.debug(f"Приклад першої трансформації: {first}")
        else:
            logger.debug("Історія трансформацій порожня")

    except Exception as e:
        logger.error(f"Помилка при експорті метаданих у JSON: {e}")
        raise

    return FileResponse(filename, media_type="application/json", filename=filename)


@router.post("/upload", response_class=HTMLResponse)
async def upload_csv(
    request: Request,
    file: UploadFile = File(...)
):
    """
        Ендпоінт для завантаження CSV файлу, аналізу даних та формування
        попереднього перегляду, метрик і графіків.
        """

    logger.info(">>> ➡️ Виклик ендпоінта /upload")

    # 1. Перевірка формату файлу
    if not file.filename.endswith(".csv"):
        logger.error("Файл не є CSV")
        raise HTTPException(status_code=400, detail="Потрібен CSV файл")
    logger.info(f"✅ Завантажено файл: {file.filename}")

    # 2. Читання даних у DataFrame
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    basic_state["dataframe"] = df
    logger.info("✅ DataFrame створено")
    logger.debug(f"Форма: {df.shape}, Колонки: {df.columns.tolist()}")

    # 3. Прев’ю перших 10 рядків
    preview = df.head(10)
    logger.debug(f"Прев’ю:\n{preview}")

    # 4. Типи даних
    columns = df.columns.tolist()
    dtypes = {col: str(df[col].dtype) for col in columns}
    logger.debug(f"Типи даних: {dtypes}")

    # 5. Підрахунок пропусків
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    logger.debug(f"Пропуски:\n{missing}")
    logger.debug(f"Відсоток пропусків:\n{missing_percent}")

    missing_info = [
        {"column": col, "missing": int(missing[col]), "percent": round(missing_percent[col], 2)}
        for col in columns
    ]
    logger.debug(f"missing_info: {missing_info}")

    missing_table = [
        {
            "column": col,
            "dtype": dtypes[col],
            "missing": int(missing[col]),
            "percent": round(missing_percent[col], 2),
            "method": None
        }
        for col in columns
    ]
    logger.debug(f"missing_table: {missing_table}")

    # 6. Загальна характеристика (info + describe)
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    logger.debug(f"df.info():\n{info_str}")

    describe_df = df.describe(include="all").transpose().reset_index()
    describe_info = describe_df.to_dict(orient="records")
    logger.debug(f"df.describe():\n{describe_info}")

    # 7. Автоматичні графіки
    plots = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=False, ax=ax)
        ax.set_title(f"Гістограма {col}")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plots[col] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
    logger.info("✅ Побудовано гістограми для числових колонок")

    for col in df.select_dtypes(include=["object", "category"]).columns:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Розподіл {col}")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plots[col] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
    logger.info("✅ Побудовано графіки для категоріальних колонок")

    # 8. Кодування та масштабування
    encoding_table = []
    for col in columns:
        dtype = dtypes[col]
        if pd.api.types.is_numeric_dtype(df[col]):
            methods = ["normalization", "standard_scaler"]
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(
                df[col]) or pd.api.types.is_string_dtype(df[col]):
            methods = ["one_hot", "label"]
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            methods = ["year", "month", "day", "timestamp", "days_diff"]
        else:
            methods = []

        encoding_table.append({
            "column": col,
            "dtype": dtype,
            "methods": methods,
            "selected": None
        })
    logger.info("✅ Сформовано таблицю кодування та масштабування")

    # 9. Кореляційна матриця
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Кореляційна матриця")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    corr_img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    logger.info("✅ Побудовано кореляційну матрицю")

    # 10. Збереження у глобальний словник
    global dataset_info
    dataset_info = {
        "dataset_name": file.filename,
        "columns": columns,
        "dtypes": dtypes,
        "missing": missing_info,
        "info": info_str,
        "describe": describe_info,
        "plots": plots
    }
    logger.info("✅ dataset_info сформовано")

    # 11. Повернення результату у шаблон
    logger.info(">>> ➡️ Передаємо дані у шаблон manual_mode_index.html")
    return templates.TemplateResponse(
        "professional/manual_mode_index.html",
        {
            "request": request,
            "table": preview.to_dict(orient="records"),
            "columns": columns,
            "dtypes": dtypes,
            "missing": missing_table,
            "info": info_str,
            "describe": describe_info,
            "plots": plots,
            "encoding_table": encoding_table
        }
    )

# ============================================================
# 🔹 Побудова графіків для аналізу даних
# ============================================================
@router.get("/analysis/plot")
async def analysis_plot(column: str = Query(...), type: str = Query(...)):
    """
    Генерує графік для вибраної колонки залежно від типу:
    - histogram: гістограма
    - boxplot: boxplot
    - heatmap: кореляційна матриця
    """
    logger.info(f">>> Виклик ендпоінта /analysis/plot з параметрами column={column}, type={type}")

    df = basic_state.get("dataframe")
    if df is None:
        logger.error("Датасет не завантажено")
        raise HTTPException(status_code=400, detail="Датасет не завантажено")

    buf = io.BytesIO()

    if type == "histogram":
        plt.figure()
        sns.histplot(df[column].dropna(), kde=False)
        plt.title(f"Гістограма {column}")
        logger.info(f"Побудовано гістограму для колонки {column}")
    elif type == "boxplot":
        plt.figure()
        sns.boxplot(x=df[column].dropna())
        plt.title(f"Boxplot {column}")
        logger.info(f"Побудовано boxplot для колонки {column}")
    elif type == "heatmap":
        plt.figure(figsize=(8,6))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Кореляційна матриця")
        logger.info("Побудовано heatmap для кореляційної матриці")

    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    logger.debug("Графік успішно закодовано у base64")
    return {"img": img_base64}


# ============================================================
# 🔹 Застосування методів обробки пропусків
# ============================================================
@router.post("/apply_missing")
async def apply_missing(methods: dict):
    """
    Застосовує вибрані методи обробки пропусків для кожної колонки:
    - drop: видалення рядків з пропусками
    - unknown: заміна пропусків на 'Unknown'
    - mean/median/mode: заповнення статистичними значеннями
    """
    logger.info(">>> Виклик ендпоінта /apply_missing")

    df = basic_state.get("dataframe")
    if df is None:
        logger.error("Датасет не завантажено")
        raise HTTPException(status_code=400, detail="Датасет не завантажено")

    for col, method in methods.items():
        logger.info(f"Обробка колонки {col} методом {method}")
        if method == "drop":
            df = df[df[col].notna()]
            log_transformation("Заповнення пропусків", col, "drop")
        elif method == "unknown":
            df[col] = df[col].fillna("Unknown")
            log_transformation("Заповнення пропусків", col, "unknown")
        elif method == "mean":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
                log_transformation("Заповнення пропусків", col, "mean")
        elif method == "median":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
                log_transformation("Заповнення пропусків", col, "median")
        elif method == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
            log_transformation("Заповнення пропусків", col, "mode")

    # Оновлюємо стан
    basic_state["dataframe"] = df
    logger.info("✅ DataFrame оновлено після обробки пропусків")

    # Повертаємо перші рядки для оновлення таблиці на фронтенді
    preview = df.head(20).replace({np.nan: None}).to_dict(orient="records")
    columns = df.columns.tolist()
    logger.debug(f"Повертаємо {len(preview)} рядків для оновлення таблиці")

    return {"columns": columns, "table": preview}


# ============================================================
# 🔹 Застосування методів кодування та масштабування
# ============================================================
@router.post("/apply_encoding")
async def apply_encoding(methods: dict):
    """
    Застосовує вибрані методи кодування/масштабування для колонок:
    - normalization / standard_scaler
    - one_hot / label
    - year / month / day / timestamp / days_diff
    """
    logger.info(">>> Виклик ендпоінта /apply_encoding")

    df = basic_state.get("dataframe")
    if df is None:
        logger.error("Датасет не завантажено")
        raise HTTPException(status_code=400, detail="Датасет не завантажено")

    for col, method in methods.items():
        logger.info(f"Обробка колонки {col} методом {method}")
        if method == "normalization":
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            log_transformation("Кодування", col, "normalization")
        elif method == "standard_scaler":
            df[col] = (df[col] - df[col].mean()) / df[col].std()
            log_transformation("Кодування", col, "standard_scaler")
        elif method == "one_hot":
            df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int)
            log_transformation("Кодування", col, "one_hot")
        elif method == "label":
            df[col] = df[col].astype("category").cat.codes
            log_transformation("Кодування", col, "label")
        elif method == "year":
            df[col+"_year"] = pd.to_datetime(df[col]).dt.year
            log_transformation("Кодування", col, "extract year")
        elif method == "month":
            df[col+"_month"] = pd.to_datetime(df[col]).dt.month
            log_transformation("Кодування", col, "extract month")
        elif method == "day":
            df[col+"_day"] = pd.to_datetime(df[col]).dt.day
            log_transformation("Кодування", col, "extract day")
        elif method == "timestamp":
            df[col+"_ts"] = pd.to_datetime(df[col]).astype(int) // 10**9
            log_transformation("Кодування", col, "timestamp")
        elif method == "days_diff":
            df[col+"_days_diff"] = (pd.to_datetime("today") - pd.to_datetime(df[col])).dt.days
            log_transformation("Кодування", col, "days_diff")
        else:
            logger.warning(f"Метод {method} не підтримується для колонки {col}")

    basic_state["dataframe"] = df
    logger.info("✅ DataFrame оновлено після кодування")

    preview = df.head(20).replace({np.nan: None}).to_dict(orient="records")
    columns = df.columns.tolist()
    logger.debug(f"Повертаємо {len(preview)} рядків після кодування")

    return {"columns": columns, "table": preview}


# ============================================================
# 🔹 Отримання списку колонок датасету
# ============================================================
@router.get("/get_columns")
async def get_columns():
    """
    Повертає список колонок поточного датасету.
    """
    logger.info(">>> Виклик ендпоінта /get_columns")

    df = basic_state.get("dataframe")
    if df is None:
        logger.error("Датасет не завантажено")
        raise HTTPException(status_code=400, detail="Датасет не завантажено")

    columns = df.columns.tolist()
    logger.debug(f"Повертаємо {len(columns)} колонок")
    return {"columns": columns}


# ============================================================
# 🔹 Побудова кореляційної матриці
# ============================================================
@router.get("/correlation_matrix")
async def correlation_matrix():
    """
    Генерує теплову карту кореляційної матриці для числових колонок.
    """
    logger.info(">>> Виклик ендпоінта /correlation_matrix")

    df = basic_state.get("dataframe")
    if df is None:
        logger.error("Датасет не завантажено")
        raise HTTPException(status_code=400, detail="Датасет не завантажено")

    buf = io.BytesIO()

    corr = df.corr(numeric_only=True)
    logger.debug("Обчислено кореляційну матрицю")

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Кореляційна матриця")

    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    logger.info("✅ Побудовано теплову карту кореляційної матриці")
    return {"img": img_base64}


# ============================================================
# 🔹 Кореляції вибраних ознак з усіма іншими
# ============================================================
@router.post("/correlation_selected")
async def correlation_selected(columns: list[str]):
    """
    Будує теплову карту кореляцій для вибраних ознак з усіма іншими.
    Повертає зображення та відсортовані значення кореляцій.
    """
    logger.info(">>> Виклик ендпоінта /correlation_selected")

    df = basic_state.get("dataframe")
    if df is None:
        logger.error("Датасет не завантажено")
        raise HTTPException(status_code=400, detail="Датасет не завантажено")

    if not columns:
        logger.error("Не вибрано жодної ознаки")
        raise HTTPException(status_code=400, detail="Не вибрано жодної ознаки")

    corr = df.corr(numeric_only=True)[columns]
    logger.debug(f"Обчислено кореляції для ознак: {columns}")

    fig, ax = plt.subplots(figsize=(10, len(df.columns) * 0.5))
    sns.heatmap(corr.T, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Кореляції вибраних ознак з усіма іншими")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    logger.info("✅ Побудовано теплову карту кореляцій")

    sorted_corr = {}
    for col in columns:
        sorted_corr[col] = sorted(corr[col].items(), key=lambda x: x[1])
    logger.debug("Відсортовано кореляції для таблиці")

    return {
        "columns": df.columns.tolist(),
        "selected": columns,
        "img": img_base64,
        "correlations": sorted_corr
    }


# ============================================================
# 🔹 Вибір ознак X та Y
# ============================================================
@router.post("/apply_xy")
async def apply_xy(selection: dict):
    """
    Зберігає вибрані ознаки X та Y у глобальний стан
    та повертає прев’ю таблиці.
    """
    logger.info(">>> Виклик ендпоінта /apply_xy")

    df = basic_state.get("dataframe")
    if df is None:
        logger.error("Датасет не завантажено")
        raise HTTPException(status_code=400, detail="Датасет не завантажено")

    X = selection.get("X", [])
    Y = selection.get("Y", [])
    logger.debug(f"Вибрано X={X[:5]}, Y={Y[:5]}")

    basic_state["X"] = X
    basic_state["Y"] = Y

    log_transformation(
        action="Вибір ознак X та Y",
        column=f"X={','.join(X)}; Y={','.join(Y)}",
        method="manual selection"
    )
    logger.info("✅ Ознаки X та Y збережено у глобальний стан")

    selected_cols = X + Y
    preview = df[selected_cols].head(20).replace({np.nan: None}).to_dict(orient="records")

    return {"X": X, "Y": Y, "table": preview}


# ============================================================
# 🔹 Розділення на Train / Validation / Test
# ============================================================
@router.post("/train_test_split")
async def train_test_split_endpoint(params: dict):
    """
    Виконує розділення датасету на train, validation та test вибірки.
    Повертає технічну інформацію та прев’ю даних.
    """
    logger.info(">>> Виклик ендпоінта /train_test_split")

    df = basic_state.get("dataframe")
    X_cols = basic_state.get("X", [])
    Y_cols = basic_state.get("Y", [])

    test_size = float(params.get("test_size", 0.2))
    val_size = float(params.get("val_size", 0.1))
    random_state = int(params.get("random_state", 42))
    logger.debug(f"Параметри розділення: test_size={test_size}, val_size={val_size}, random_state={random_state}")

    X = df[X_cols]
    Y = df[Y_cols]

    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=val_relative_size, random_state=random_state)

    basic_state.update({
        "x_train": X_train, "x_val": X_val, "x_test": X_test,
        "y_train": Y_train, "y_val": Y_val, "y_test": Y_test
    })
    log_transformation("Train/Test/Validation Split", None,
                       f"test_size={test_size}, val_size={val_size}, random_state={random_state}")
    logger.info("✅ Виконано розділення на train/val/test")

    logger.debug(f"Форми: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}, "
                 f"Y_train={Y_train.shape}, Y_val={Y_val.shape}, Y_test={Y_test.shape}")

    preview = lambda df: df.head(10).replace({np.nan: None}).to_dict(orient="records")

    return {
        "X_columns": X_cols,
        "Y_columns": Y_cols,
        "shapes": {
            "x_train": X_train.shape,
            "x_val": X_val.shape,
            "x_test": X_test.shape,
            "y_train": Y_train.shape,
            "y_val": Y_val.shape,
            "y_test": Y_test.shape
        },
        "x_train": preview(X_train),
        "x_val": preview(X_val),
        "x_test": preview(X_test),
        "y_train": preview(Y_train),
        "y_val": preview(Y_val),
        "y_test": preview(Y_test)
    }

# ============================================================
# 🔹 Завантаження окремої частини датасету (Train/Val/Test)
# ============================================================
@router.get("/download_split/{part}")
async def download_split(part: str):
    """
    Завантажує окрему частину датасету у форматі CSV.
    Доступні частини: x_train, x_val, x_test, y_train, y_val, y_test.
    """
    logger.info(f">>> Виклик ендпоінта /download_split для {part}")

    if part not in ["x_train", "x_val", "x_test", "y_train", "y_val", "y_test"]:
        logger.error("Невірна частина датасету")
        raise HTTPException(status_code=400, detail="Невірна частина датасету")

    df_part = basic_state.get(part)
    if df_part is None:
        logger.error(f"Частина {part} не знайдена")
        raise HTTPException(status_code=400, detail="Частина датасету не знайдена")

    filename = f"{part}.csv"
    df_part.to_csv(filename, index=False)
    logger.info(f"✅ Збережено {filename} ({df_part.shape[0]} рядків, {df_part.shape[1]} колонок)")
    return FileResponse(filename, media_type="text/csv", filename=filename)


# ============================================================
# 🔹 Завантаження Train/Val/Test у ZIP
# ============================================================
@router.get("/download_split_zip")
async def download_split_zip():
    """
    Завантажує всі частини датасету (Train/Val/Test) у ZIP архів.
    Додає README.txt з інформацією про розміри вибірок.
    """
    logger.info(">>> Виклик ендпоінта /download_split_zip")

    parts = {
        "x_train.csv": basic_state.get("x_train"),
        "x_val.csv": basic_state.get("x_val"),
        "x_test.csv": basic_state.get("x_test"),
        "y_train.csv": basic_state.get("y_train"),
        "y_val.csv": basic_state.get("y_val"),
        "y_test.csv": basic_state.get("y_test"),
    }

    if any(df is None for df in parts.values()):
        logger.error("Не всі частини датасету доступні")
        raise HTTPException(status_code=400, detail="Спочатку виконайте train/test/validation split")

    zip_filename = "train_val_test_split.zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for fname, df in parts.items():
            df.to_csv(fname, index=False)
            zipf.write(fname)
            os.remove(fname)
            logger.debug(f"Додано {fname} у архів ({df.shape[0]}×{df.shape[1]})")

        with open("README.txt", "w", encoding="utf-8") as f:
            f.write("Розміри вибірок:\n")
            for key, df in parts.items():
                f.write(f"{key}: {df.shape[0]} × {df.shape[1]}\n")
        zipf.write("README.txt")
        os.remove("README.txt")

    logger.info("✅ ZIP архів створено")
    return FileResponse(zip_filename, media_type="application/zip", filename=zip_filename)


# ============================================================
# 🔹 Завантаження всього обробленого датасету
# ============================================================
@router.get("/download_full_dataset")
async def download_full_dataset():
    """
    Завантажує весь оброблений датасет у форматі CSV.
    """
    logger.info(">>> Виклик ендпоінта /download_full_dataset")

    df = basic_state.get("dataframe")
    if df is None:
        logger.error("Датасет не завантажено")
        raise HTTPException(status_code=400, detail="Датасет не завантажено")

    filename = "processed_dataset.csv"
    df.to_csv(filename, index=False)
    logger.info(f"✅ Збережено {filename} ({df.shape[0]} рядків, {df.shape[1]} колонок)")
    return FileResponse(filename, media_type="text/csv", filename=filename)


# ============================================================
# 🔹 Завантаження X та Y у ZIP
# ============================================================
@router.get("/download_xy_zip")
async def download_xy_zip():
    """
    Завантажує вибрані ознаки X та Y у ZIP архів.
    """
    logger.info(">>> Виклик ендпоінта /download_xy_zip")

    df = basic_state.get("dataframe")
    X_cols = basic_state.get("X", [])
    Y_cols = basic_state.get("Y", [])

    if df is None or not X_cols or not Y_cols:
        logger.error("Не вибрано X та Y ознаки або датасет не завантажено")
        raise HTTPException(status_code=400, detail="Не вибрано X та Y ознаки")

    x_df = df[X_cols]
    y_df = df[Y_cols]

    filenames = {"X.csv": x_df, "Y.csv": y_df}
    zip_filename = "XY_split.zip"

    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for fname, part_df in filenames.items():
            part_df.to_csv(fname, index=False)
            zipf.write(fname)
            os.remove(fname)
            logger.debug(f"Додано {fname} у архів ({part_df.shape[0]}×{part_df.shape[1]})")

    logger.info("✅ ZIP архів X/Y створено")
    return FileResponse(zip_filename, media_type="application/zip", filename=zip_filename)


# ============================================================
# 🔹 Інформація про Y ознаки
# ============================================================
@router.get("/get_y_info")
async def get_y_info():
    """
    Повертає список вибраних Y ознак та їх типи даних.
    """
    logger.info(">>> Виклик ендпоінта /get_y_info")

    df = basic_state.get("dataframe")
    Y_cols = basic_state.get("Y", [])
    if df is None:
        logger.error("Датасет не завантажено")
        raise HTTPException(status_code=400, detail="Датасет не завантажено")

    Y_dtypes = [str(df[col].dtype) for col in Y_cols] if Y_cols else []
    logger.debug(f"Y ознаки: {Y_cols}, типи: {Y_dtypes}")
    return {"Y": Y_cols, "Y_dtypes": Y_dtypes}


# ============================================================
# 🔹 Вибір типу задачі
# ============================================================
@router.post("/start_task")
async def start_task(payload: dict):
    """
    Зберігає вибраний тип задачі у глобальний стан.
    """
    logger.info(">>> Виклик ендпоінта /start_task")

    task_type = payload.get("task_type")
    basic_state["task_type"] = task_type

    log_transformation("Вибір типу задачі", None, task_type)
    logger.info(f"✅ Тип задачі встановлено: {task_type}")

    return {"status": "ok", "task_type": task_type}


# ============================================================
# 🔹 Перетворення numpy типів у стандартні Python
# ============================================================
def to_native(obj):
    """
    Перетворює numpy типи у стандартні Python:
    - np.integer → int
    - np.floating → float
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    return obj


# ============================================================
# 🔹 Функція для запуску регресії з Train/Val/Test
# ============================================================
def run_regression_models_random(X_train, X_val, X_test, Y_train, Y_val, Y_test, n_iter=20):
    """
    Запускає кілька моделей регресії з випадковим пошуком гіперпараметрів.
    Повертає топ-3 моделі за метрикою R² з їх параметрами, метриками та графіками
    для валідаційної та (за наявності) тестової вибірки.
    """
    logger.info(">>> Старт функції run_regression_models_random")

    results = []

    models = {
        "LinearRegression": (LinearRegression(), {}),
        "RandomForestRegressor": (RandomForestRegressor(), {
            "n_estimators": np.arange(50, 300, 50),
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }),
        "GradientBoostingRegressor": (GradientBoostingRegressor(), {
            "n_estimators": np.arange(50, 300, 50),
            "learning_rate": np.linspace(0.01, 0.2, 10),
            "max_depth": [3, 5, 7]
        }),
        "XGBRegressor": (xgb.XGBRegressor(objective="reg:squarederror"), {
            "n_estimators": np.arange(50, 300, 50),
            "learning_rate": np.linspace(0.01, 0.2, 10),
            "max_depth": [3, 5, 7, 10],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        })
    }

    for name, (model, param_dist) in models.items():
        logger.info(f"Запуск моделі {name}")

        # 🔹 RandomizedSearchCV
        if param_dist:
            search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=3,
                scoring="r2",
                n_jobs=-1,
                random_state=42,
                error_score="raise"
            )
            search.fit(X_train, Y_train.values.ravel())
            best_model = search.best_estimator_
            best_params = {k: to_native(v) for k, v in search.best_params_.items()}
        else:
            model.fit(X_train, Y_train.values.ravel())
            best_model = model
            best_params = {}

        # --- Метрики на валідації ---
        y_pred_val = best_model.predict(X_val)
        val_metrics = {
            "r2": r2_score(Y_val, y_pred_val),
            "mae": mean_absolute_error(Y_val, y_pred_val),
            "rmse": np.sqrt(mean_squared_error(Y_val, y_pred_val)),
            "explained_variance": explained_variance_score(Y_val, y_pred_val)
        }

        # --- Метрики на тесті (якщо є) ---
        test_metrics = None
        if X_test is not None and Y_test is not None:
            y_pred_test = best_model.predict(X_test)
            test_metrics = {
                "r2": r2_score(Y_test, y_pred_test),
                "mae": mean_absolute_error(Y_test, y_pred_test),
                "rmse": np.sqrt(mean_squared_error(Y_test, y_pred_test)),
                "explained_variance": explained_variance_score(Y_test, y_pred_test)
            }
            logger.info(f"Метрики {name} на тесті: {test_metrics}")

        # --- Графіки ---
        plots = {}

        # Scatter plot (Val)
        fig, ax = plt.subplots()
        sns.scatterplot(x=Y_val.values.ravel(), y=y_pred_val, ax=ax)
        ax.set_title(f"{name} — Scatter (Val)")
        buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
        plots["scatter_val"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        # Residual plot (Val)
        residuals_val = Y_val.values.ravel() - y_pred_val
        fig, ax = plt.subplots()
        ax.scatter(y_pred_val, residuals_val, alpha=0.6, color="red")
        ax.axhline(y=0, color="black", linestyle="--")
        ax.set_title(f"{name} — Residuals (Val)")
        buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
        plots["residuals_val"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        # Q-Q plot (Val)
        fig, ax = plt.subplots()
        stats.probplot(residuals_val, dist="norm", plot=ax)
        ax.set_title(f"{name} — Q-Q (Val)")
        buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
        plots["qq_val"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        # --- Графіки для тесту (якщо є) ---
        if X_test is not None and Y_test is not None:
            y_pred_test = best_model.predict(X_test)
            residuals_test = Y_test.values.ravel() - y_pred_test

            # Scatter plot (Test)
            fig, ax = plt.subplots()
            sns.scatterplot(x=Y_test.values.ravel(), y=y_pred_test, ax=ax)
            ax.set_title(f"{name} — Scatter (Test)")
            buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
            plots["scatter_test"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

            # Residual plot (Test)
            fig, ax = plt.subplots()
            ax.scatter(y_pred_test, residuals_test, alpha=0.6, color="blue")
            ax.axhline(y=0, color="black", linestyle="--")
            ax.set_title(f"{name} — Residuals (Test)")
            buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
            plots["residuals_test"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

            # Q-Q plot (Test)
            fig, ax = plt.subplots()
            stats.probplot(residuals_test, dist="norm", plot=ax)
            ax.set_title(f"{name} — Q-Q (Test)")
            buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
            plots["qq_test"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

        results.append({
            "model": name,
            "best_params": best_params,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "plots": plots
        })

    logger.info("✅ Завершено роботу run_regression_models_random")
    return sorted(results, key=lambda x: x["val_metrics"]["r2"], reverse=True)[:3]



# ============================================================
# 🔹 Функція для запуску класифікації
# ============================================================
def run_classification_models_random(X_train, X_val, X_test, Y_train, Y_val, Y_test, n_iter=20):
    """
    Запускає кілька моделей класифікації з випадковим пошуком гіперпараметрів.
    Повертає топ-3 моделі за метрикою Accuracy з їх параметрами, метриками та графіками
    для валідаційної та (за наявності) тестової вибірки.
    """
    logger.info(">>> Старт функції run_classification_models_random")

    results = []
    n_classes = len(np.unique(Y_train))
    solvers = ["lbfgs", "saga", "newton-cg"] if n_classes > 2 else ["lbfgs", "liblinear", "saga", "newton-cg"]

    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {
            "C": np.logspace(-2, 2, 10),
            "solver": solvers
        }),
        "DecisionTreeClassifier": (DecisionTreeClassifier(), {
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10]
        }),
        "RandomForestClassifier": (RandomForestClassifier(), {
            "n_estimators": np.arange(50, 300, 50),
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10]
        }),
        "SVM": (SVC(probability=True), {
            "C": np.logspace(-2, 2, 10),
            "kernel": ["linear", "rbf", "poly"]
        })
    }

    for name, (model, param_dist) in models.items():
        logger.info(f"Запуск моделі {name}")

        # 🔹 RandomizedSearchCV
        if param_dist:
            search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=min(n_iter, len(param_dist)),
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                random_state=42,
                error_score="raise"
            )
            search.fit(X_train, Y_train.values.ravel())
            best_model = search.best_estimator_
            best_params = {k: to_native(v) for k, v in search.best_params_.items()}
        else:
            model.fit(X_train, Y_train.values.ravel())
            best_model = model
            best_params = {}

        # --- Метрики на валідації ---
        y_pred_val = best_model.predict(X_val)
        val_metrics = {
            "accuracy": accuracy_score(Y_val, y_pred_val),
            "precision": precision_score(Y_val, y_pred_val, average="weighted", zero_division=0),
            "recall": recall_score(Y_val, y_pred_val, average="weighted", zero_division=0),
            "f1": f1_score(Y_val, y_pred_val, average="weighted", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(Y_val, y_pred_val),
            "mcc": matthews_corrcoef(Y_val, y_pred_val),
            "kappa": cohen_kappa_score(Y_val, y_pred_val)
        }

        # --- Метрики на тесті (якщо є) ---
        test_metrics = None
        if X_test is not None and Y_test is not None:
            y_pred_test = best_model.predict(X_test)
            test_metrics = {
                "accuracy": accuracy_score(Y_test, y_pred_test),
                "precision": precision_score(Y_test, y_pred_test, average="weighted", zero_division=0),
                "recall": recall_score(Y_test, y_pred_test, average="weighted", zero_division=0),
                "f1": f1_score(Y_test, y_pred_test, average="weighted", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(Y_test, y_pred_test),
                "mcc": matthews_corrcoef(Y_test, y_pred_test),
                "kappa": cohen_kappa_score(Y_test, y_pred_test)
            }
            logger.info(f"Метрики {name} на тесті: {test_metrics}")

        # --- Графіки ---
        plots = {}

        # --- Метрики на тесті (якщо є) ---
        test_metrics = None
        if X_test is not None and Y_test is not None:
            y_pred_test = best_model.predict(X_test)
            test_metrics = {
                "accuracy": accuracy_score(Y_test, y_pred_test),
                "precision": precision_score(Y_test, y_pred_test, average="weighted", zero_division=0),
                "recall": recall_score(Y_test, y_pred_test, average="weighted", zero_division=0),
                "f1": f1_score(Y_test, y_pred_test, average="weighted", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(Y_test, y_pred_test),
                "mcc": matthews_corrcoef(Y_test, y_pred_test),
                "kappa": cohen_kappa_score(Y_test, y_pred_test)
            }
            logger.info(f"Метрики {name} на тесті: {test_metrics}")

        # --- Графіки ---
        plots = {}

        # Confusion Matrix (Val)
        cm_val = confusion_matrix(Y_val, y_pred_val)
        fig, ax = plt.subplots()
        sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{name} — Матриця неточностей (Val)")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plots["confusion_matrix_val"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        # Confusion Matrix (Test)
        if X_test is not None and Y_test is not None:
            cm_test = confusion_matrix(Y_test, y_pred_test)
            fig, ax = plt.subplots()
            sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", ax=ax)
            ax.set_title(f"{name} — Матриця неточностей (Test)")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plots["confusion_matrix_test"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

        # ROC/PR криві — теж варто розділити на Val/Test
        if hasattr(best_model, "predict_proba"):
            y_score_val = best_model.predict_proba(X_val)
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(Y_val, y_score_val[:, 1])
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label="ROC curve (Val)")
                ax.plot([0, 1], [0, 1], "k--")
                ax.set_title(f"{name} — ROC‑крива (Val)")
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                plots["roc_curve_val"] = base64.b64encode(buf.read()).decode("utf-8")
                plt.close(fig)

                precision, recall, _ = precision_recall_curve(Y_val, y_score_val[:, 1])
                fig, ax = plt.subplots()
                ax.plot(recall, precision, label="PR curve (Val)")
                ax.set_title(f"{name} — Precision‑Recall крива (Val)")
                buf = io.BytesIO();
                plt.savefig(buf, format="png")
                buf.seek(0)
                plots["pr_curve_val"] = base64.b64encode(buf.read()).decode("utf-8")
                plt.close(fig)

            # Для тесту
            if X_test is not None and Y_test is not None:
                y_score_test = best_model.predict_proba(X_test)
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(Y_test, y_score_test[:, 1])
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label="ROC curve (Test)")
                    ax.plot([0, 1], [0, 1], "k--")
                    ax.set_title(f"{name} — ROC‑крива (Test)")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    plots["roc_curve_test"] = base64.b64encode(buf.read()).decode("utf-8")
                    plt.close(fig)

                    precision, recall, _ = precision_recall_curve(Y_test, y_score_test[:, 1])
                    fig, ax = plt.subplots()
                    ax.plot(recall, precision, label="PR curve (Test)")
                    ax.set_title(f"{name} — Precision‑Recall крива (Test)")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    plots["pr_curve_test"] = base64.b64encode(buf.read()).decode("utf-8")
                    plt.close(fig)

        results.append({
            "model": name,
            "best_params": best_params,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "plots": plots
        })

    logger.info("✅ Завершено роботу run_classification_models_random")
    return sorted(results, key=lambda x: x["val_metrics"]["accuracy"], reverse=True)[:3]

# ============================================================
# 🔹 Функція для запуску кластеризації з цільовою змінною
# ============================================================
def run_clustering_models(X_train, Y_train, X_val=None, Y_val=None, X_test=None, Y_test=None):
    """
        Запускає кілька алгоритмів кластеризації (KMeans, DBSCAN, SpectralClustering, GaussianMixture),
        обчислює метрики та будує графіки. Повертає результати у вигляді списку словників.
        """
    logger.info(">>> Старт функції run_clustering_models")

    # 🔹 Перетворення Y у 1D
    if isinstance(Y_train, np.ndarray) and Y_train.ndim > 1:
        Y_train = Y_train.ravel()
    elif hasattr(Y_train, "values"):
        Y_train = Y_train.values.ravel()

    if Y_val is not None:
        if hasattr(Y_val, "values"):
            Y_val = Y_val.values.ravel()
        elif isinstance(Y_val, np.ndarray) and Y_val.ndim > 1:
            Y_val = Y_val.ravel()

    if Y_test is not None:
        if hasattr(Y_test, "values"):
            Y_test = Y_test.values.ravel()
        elif isinstance(Y_test, np.ndarray) and Y_test.ndim > 1:
            Y_test = Y_test.ravel()

    results = []

    models = {
        "KMeans": KMeans(n_clusters=len(np.unique(Y_train)), random_state=42, max_iter=50, n_init=5),
        "DBSCAN": None,  # DBSCAN будемо запускати окремо з циклом
        "SpectralClustering": SpectralClustering(n_clusters=len(np.unique(Y_train)), random_state=42, affinity="nearest_neighbors"),
        "GaussianMixture": GaussianMixture(n_components=len(np.unique(Y_train)), random_state=42, max_iter=50)
    }

    for name, model in models.items():
        logger.info(f"\n=== Запуск моделі {name} ===")

        # --- TRAIN ---
        if name == "GaussianMixture":
            model.fit(X_train)
            clusters_train = model.predict(X_train)
            logger.debug(f"[{name}] Log-likelihood={model.lower_bound_:.4f}, Ітерацій={model.n_iter_}")

        elif name == "DBSCAN":
            best_sil = -1
            best_params = None
            best_clusters = None
            best_model = None

            for eps in [0.3, 0.5, 0.7, 1.0]:
                for min_samples in [3, 5, 10]:
                    model_tmp = DBSCAN(eps=eps, min_samples=min_samples)
                    clusters_tmp = model_tmp.fit_predict(X_train)
                    n_clusters = len(set(clusters_tmp)) - (1 if -1 in clusters_tmp else 0)
                    noise_points = sum(clusters_tmp == -1)
                    logger.debug(
                        f"[DBSCAN] eps={eps}, min_samples={min_samples} → кластерів={n_clusters}, шумових={noise_points}")

                    if n_clusters > 1:
                        sil_tmp = silhouette_score(X_train, clusters_tmp)
                        logger.debug(f"[DBSCAN] Silhouette={sil_tmp:.3f}")

                        if sil_tmp > best_sil:
                            best_sil = sil_tmp
                            best_params = (eps, min_samples)
                            best_clusters = clusters_tmp
                            best_model = model_tmp

            if best_clusters is not None:
                clusters_train = best_clusters
                model = best_model  # зберігаємо найкращу модель для валідації/тесту
                logger.info(
                    f"[DBSCAN] ✅ Найкращі параметри: eps={best_params[0]}, min_samples={best_params[1]}, Silhouette={best_sil:.3f}")

            else:
                model = DBSCAN(eps=0.5, min_samples=5)
                clusters_train = model.fit_predict(X_train)
                logger.debug("[DBSCAN] ⚠️ Усі варіанти дали лише один кластер (шум)")

        else:
            clusters_train = model.fit_predict(X_train)
            if name == "KMeans":
                logger.debug(f"[{name}] Inertia: {model.inertia_:.4f}")
                logger.debug(f"[{name}] Кількість ітерацій: {model.n_iter_}")
            elif name == "SpectralClustering":
                logger.debug(f"[{name}] Виконано спектральну кластеризацію")

        # Розподіл кластерів
        unique, counts = np.unique(clusters_train, return_counts=True)
        cluster_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
        logger.debug(f"[{name}] Розподіл кластерів (train): {cluster_distribution}")

        # Метрики без Y (train)
        if len(set(clusters_train)) > 1:
            sil = silhouette_score(X_train, clusters_train)
            ch = calinski_harabasz_score(X_train, clusters_train)
            db = davies_bouldin_score(X_train, clusters_train)
            logger.debug(f"[{name}] Train Silhouette={sil:.3f}, CH={ch:.3f}, DB={db:.3f}")
        else:
            sil, ch, db = None, None, None
            logger.debug(f"[{name}] ⚠️ Неможливо обчислити Silhouette/CH/DB — лише один кластер")

        # Метрики з Y (train)
        ari = adjusted_rand_score(Y_train, clusters_train)
        nmi = normalized_mutual_info_score(Y_train, clusters_train)
        homogeneity = homogeneity_score(Y_train, clusters_train)
        logger.debug(f"[{name}] Train ARI={ari:.3f}, NMI={nmi:.3f}, Homogeneity={homogeneity:.3f}")

        # --- VALIDATION ---
        if X_val is not None:
            clusters_val = model.fit_predict(X_val) if name != "GaussianMixture" else model.predict(X_val)
            if len(set(clusters_val)) > 1:
                sil_val = silhouette_score(X_val, clusters_val)
                logger.debug(f"[{name}] Validation Silhouette={sil_val:.3f}")
            else:
                logger.debug(f"[{name}] ⚠️ Validation Silhouette не обчислюється — лише один кластер")
            if Y_val is not None:
                ari_val = adjusted_rand_score(Y_val, clusters_val)
                logger.debug(f"[{name}] Validation ARI={ari_val:.3f}")

        # --- TEST ---
        if X_test is not None:
            clusters_test = model.fit_predict(X_test) if name != "GaussianMixture" else model.predict(X_test)
            if len(set(clusters_test)) > 1:
                sil_test = silhouette_score(X_test, clusters_test)
                logger.debug(f"[{name}] Test Silhouette={sil_test:.3f}")
            else:
                logger.debug(f"[{name}] ⚠️ Test Silhouette не обчислюється — лише один кластер")
            if Y_test is not None:
                ari_test = adjusted_rand_score(Y_test, clusters_test)
                logger.debug(f"[{name}] Test ARI={ari_test:.3f}")

        # Confusion matrix (train)
        cm = confusion_matrix(Y_train, clusters_train)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Кластери")
        ax.set_ylabel("Справжні класи")
        ax.set_title(f"{name} — Матриця неточностей (Train)")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_cm = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        # Cluster distribution bar plot
        fig, ax = plt.subplots()
        ax.bar(cluster_distribution.keys(), cluster_distribution.values(), color="skyblue")
        ax.set_title(f"{name} — Розподіл кластерів")
        ax.set_xlabel("Кластер")
        ax.set_ylabel("Кількість точок")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_dist = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        # Silhouette plot (якщо кластерів > 1)
        plot_sil = None
        if len(set(clusters_train)) > 1:
            fig, ax = plt.subplots()
            silhouette_vals = silhouette_samples(X_train, clusters_train)
            y_ticks = []
            y_lower, y_upper = 0, 0
            for i, cluster in enumerate(np.unique(clusters_train)):
                cluster_silhouette_vals = silhouette_vals[clusters_train == cluster]
                cluster_silhouette_vals.sort()
                y_upper += len(cluster_silhouette_vals)
                ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none')
                y_ticks.append((y_lower + y_upper) / 2)
                y_lower += len(cluster_silhouette_vals)
            ax.set_title(f"{name} — Графік силуетів кластерів")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_sil = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

        # PCA scatter plot
        plot_pca = None
        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_train)
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_train, cmap="tab10", alpha=0.7)
            ax.set_title(f"{name} — Візуалізація кластерів (PCA)")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_pca = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
        except Exception as e:
            print(f"[{name}] ⚠️ PCA plot error: {e}")

        results.append({
            "model": name,
            "ari": float(ari),
            "nmi": float(nmi),
            "homogeneity": float(homogeneity),
            "silhouette": float(sil) if sil is not None else None,
            "calinski_harabasz": float(ch) if ch is not None else None,
            "davies_bouldin": float(db) if db is not None else None,
            "cluster_distribution": cluster_distribution,
            "plots": {
                "confusion_matrix — Матриця неточностей (Train)": plot_cm,
                "cluster_distribution — Розподіл кластерів": plot_dist,
                "silhouette_plot — Графік силуетів кластерів": plot_sil,
                "pca_plot — Візуалізація кластерів (PCA)": plot_pca
            }
        })

    logger.info("✅ Завершено роботу run_clustering_models")
    return sorted(results, key=lambda x: x["ari"], reverse=True)


# ============================================================
# 🔹 Функція для запуску кластеризації без цільової змінної
# ============================================================
def run_no_target_clustering(X_train, X_test=None, X_val=None):
    """
    Виконує кластеризацію без цільової змінної (unsupervised).
    Запускає KMeans, DBSCAN, SpectralClustering та GaussianMixture.
    Обчислює метрики та будує графіки для train/val/test.
    """
    logger.info(">>> Старт функції run_no_target_clustering")
    results = []

    # 🔹 Оптимальна кількість кластерів для KMeans
    inertias = []
    K_range = range(2, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_train)
        inertias.append(km.inertia_)
    diffs = np.diff(inertias)
    optimal_k = K_range[np.argmin(diffs) + 1]
    logger.info(f"[KMeans] Оптимальна кількість кластерів = {optimal_k}")

    # 🔹 Моделі для запуску
    models = {
        "KMeans": KMeans(n_clusters=optimal_k, random_state=42),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "SpectralClustering": SpectralClustering(n_clusters=optimal_k, random_state=42),
        "GaussianMixture": GaussianMixture(n_components=optimal_k, random_state=42)
    }

    for name, model in models.items():
        logger.info(f"Запуск моделі {name} (No Target)")

        clusters_train = model.fit_predict(X_train) if name != "GaussianMixture" else model.fit(X_train).predict(X_train)

        # --- TRAIN метрики ---
        if len(set(clusters_train)) > 1:
            sil_train = silhouette_score(X_train, clusters_train)
            ch_train = calinski_harabasz_score(X_train, clusters_train)
            db_train = davies_bouldin_score(X_train, clusters_train)
            logger.info(f"[{name}] Train Silhouette={sil_train:.3f}, CH={ch_train:.3f}, DB={db_train:.3f}")
        else:
            sil_train, ch_train, db_train = None, None, None
            logger.warning(f"[{name}] Неможливо обчислити Silhouette/CH/DB — лише один кластер")

        # --- TRAIN графіки ---
        plots = {}
        # Scatter plot
        fig, ax = plt.subplots()
        ax.scatter(X_train[:, 0], X_train[:, 1], c=clusters_train, cmap="viridis", alpha=0.6)
        ax.set_title(f"{name} — Кластеризація (Train)")
        buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
        plots["train_scatter — Графік кластерів (Train)"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        logger.debug(f"[{name}] Побудовано scatter plot для Train")

        # Silhouette plot
        if sil_train is not None:
            fig, ax = plt.subplots()
            silhouette_vals = silhouette_samples(X_train, clusters_train)
            y_lower = 0
            for i, cluster in enumerate(np.unique(clusters_train)):
                cluster_vals = silhouette_vals[clusters_train == cluster]
                cluster_vals.sort()
                y_upper = y_lower + len(cluster_vals)
                ax.barh(range(y_lower, y_upper), cluster_vals, edgecolor='none')
                y_lower = y_upper
            ax.set_title(f"{name} — Silhouette Plot (Train)")
            buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
            plots["train_silhouette_plot — Графік силуетів (Train)"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            logger.debug(f"[{name}] Побудовано silhouette plot для Train")

        # PCA scatter plot
        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_train)
            fig, ax = plt.subplots()
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_train, cmap="tab10", alpha=0.7)
            ax.set_title(f"{name} — PCA візуалізація кластерів (Train)")
            buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
            plots["train_pca_plot — PCA візуалізація (Train)"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            logger.debug(f"[{name}] Побудовано PCA plot для Train")
        except Exception as e:
            logger.error(f"[{name}] Помилка PCA plot: {e}")

        # --- VALIDATION ---
        if X_val is not None:
            clusters_val = model.predict(X_val) if name in ["KMeans","GaussianMixture"] else model.fit_predict(X_val)
            if len(set(clusters_val)) > 1:
                sil_val = silhouette_score(X_val, clusters_val)
                ch_val = calinski_harabasz_score(X_val, clusters_val)
                db_val = davies_bouldin_score(X_val, clusters_val)
                logger.info(f"[{name}] Validation Silhouette={sil_val:.3f}, CH={ch_val:.3f}, DB={db_val:.3f}")
            else:
                sil_val, ch_val, db_val = None, None, None
                logger.warning(f"[{name}] Validation метрики не обчислюються — лише один кластер")

            # Scatter plot (Val)
            fig, ax = plt.subplots()
            ax.scatter(X_val[:, 0], X_val[:, 1], c=clusters_val, cmap="viridis", alpha=0.6)
            ax.set_title(f"{name} — Кластеризація (Val)")
            buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
            plots["val_scatter — Графік кластерів (Val)"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            logger.debug(f"[{name}] Побудовано scatter plot для Validation")
        else:
            sil_val, ch_val, db_val = None, None, None

        # --- TEST ---
        if X_test is not None and name in ["KMeans","GaussianMixture"]:
            clusters_test = model.predict(X_test)
            if len(set(clusters_test)) > 1:
                sil_test = silhouette_score(X_test, clusters_test)
                ch_test = calinski_harabasz_score(X_test, clusters_test)
                db_test = davies_bouldin_score(X_test, clusters_test)
                logger.info(f"[{name}] Test Silhouette={sil_test:.3f}, CH={ch_test:.3f}, DB={db_test:.3f}")
            else:
                sil_test, ch_test, db_test = None, None, None
                logger.warning(f"[{name}] Test метрики не обчислюються — лише один кластер")

            # Scatter plot (Test)
            fig, ax = plt.subplots()
            ax.scatter(X_test[:, 0], X_test[:, 1], c=clusters_test, cmap="viridis", alpha=0.6)
            ax.set_title(f"{name} — Кластеризація (Test)")
            buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
            plots["test_scatter — Графік кластерів (Test)"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            logger.debug(f"[{name}] Побудовано scatter plot для Test")
        else:
            sil_test, ch_test, db_test = None, None, None

        results.append({
            "model": name,
            "train_silhouette": float(sil_train) if sil_train is not None else None,
            "train_calinski_harabasz": float(ch_train) if ch_train is not None else None,
            "train_davies_bouldin": float(db_train) if db_train is not None else None,
            "val_silhouette": float(sil_val) if sil_val is not None else None,
            "val_calinski_harabasz": float(ch_val) if ch_val is not None else None,
            "val_davies_bouldin": float(db_val) if db_val is not None else None,
            "test_silhouette": float(sil_test) if sil_test is not None else None,
            "test_calinski_harabasz": float(ch_test) if ch_test is not None else None,
            "test_davies_bouldin": float(db_test) if db_test is not None else None,
            "plots": plots
        })

    logger.info("✅ Завершено роботу run_no_target_clustering")
    return results


# ============================================================
# 🔹 Функція для запуску мульти-регресії
# ============================================================
def run_multi_regression(X_train, Y_train, X_val=None, Y_val=None, X_test=None, Y_test=None):
    """
    Запускає MultiOutput регресію для кількох моделей (RandomForest, GradientBoosting).
    Обчислює метрики (MSE, R²) для train/val/test та будує графіки.
    """
    logger.info(">>> Старт функції run_multi_regression")

    results = []
    base_models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    for name, base_model in base_models.items():
        logger.info(f"Запуск моделі {name} (MultiOutput)")
        logger.debug(f"Форма Train: {getattr(X_train, 'shape', None)}, {getattr(Y_train, 'shape', None)}")
        if X_val is not None:
            logger.debug(f"Форма Val: {getattr(X_val, 'shape', None)}, {getattr(Y_val, 'shape', None)}")
        if X_test is not None:
            logger.debug(f"Форма Test: {getattr(X_test, 'shape', None)}, {getattr(Y_test, 'shape', None)}")

        model = MultiOutputRegressor(base_model)
        model.fit(X_train, Y_train)
        logger.info(f"Модель {name} навчена")

        # --- Train ---
        preds_train = model.predict(X_train)
        mse_train = mean_squared_error(Y_train, preds_train)
        r2_train = r2_score(Y_train, preds_train)
        logger.info(f"[{name}] Train MSE={mse_train:.4f}, R²={r2_train:.4f}")

        # --- Validation ---
        mse_val, r2_val, preds_val = None, None, None
        if X_val is not None and Y_val is not None:
            preds_val = model.predict(X_val)
            mse_val = mean_squared_error(Y_val, preds_val)
            r2_val = r2_score(Y_val, preds_val)
            logger.info(f"[{name}] Val MSE={mse_val:.4f}, R²={r2_val:.4f}")

        # --- Test ---
        mse_test, r2_test, preds_test = None, None, None
        if X_test is not None and Y_test is not None:
            preds_test = model.predict(X_test)
            mse_test = mean_squared_error(Y_test, preds_test)
            r2_test = r2_score(Y_test, preds_test)
            logger.info(f"[{name}] Test MSE={mse_test:.4f}, R²={r2_test:.4f}")

        # --- Графіки ---
        plots = {}

        # Scatter plot (Train)
        Y_train_np, preds_train_np = np.array(Y_train), np.array(preds_train)
        fig, ax = plt.subplots()
        ax.scatter(Y_train_np[:, 0], preds_train_np[:, 0], alpha=0.6, color="blue")
        ax.plot([Y_train_np[:, 0].min(), Y_train_np[:, 0].max()],
                [Y_train_np[:, 0].min(), Y_train_np[:, 0].max()],
                "r--", lw=2)
        ax.set_title(f"{name} (Train R²={r2_train:.3f})")
        buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
        plots["train_scatter — Графік фактичних vs передбачених (Train)"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        logger.debug(f"[{name}] Побудовано scatter plot для Train")

        # Residual plot (Validation)
        if preds_val is not None:
            residuals = np.array(Y_val) - preds_val
            fig, ax = plt.subplots()
            ax.scatter(preds_val[:, 0], residuals[:, 0], alpha=0.6, color="red")
            ax.axhline(y=0, color="black", linestyle="--")
            ax.set_title(f"{name} Residuals (Val)")
            buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
            plots["val_residuals — Графік похибок на валідації"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            logger.debug(f"[{name}] Побудовано residual plot для Validation")

        # Distribution of residuals (Test)
        if preds_test is not None:
            residuals_test = np.array(Y_test) - preds_test
            fig, ax = plt.subplots()
            sns.histplot(residuals_test[:, 0], bins=20, kde=True, ax=ax)
            ax.set_title(f"{name} Residuals Distribution (Test)")
            buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
            plots["test_residuals_distribution — Розподіл похибок на тесті"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            logger.debug(f"[{name}] Побудовано distribution plot для Test")

        # Q-Q plot (Validation)
        if preds_val is not None:
            residuals = np.array(Y_val) - preds_val
            fig, ax = plt.subplots()
            stats.probplot(residuals[:, 0], dist="norm", plot=ax)
            ax.set_title(f"{name} Q-Q Plot (Val)")
            buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
            plots["val_qq_plot — Q-Q графік похибок на валідації"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            logger.debug(f"[{name}] Побудовано Q-Q plot для Validation")

        results.append({
            "model": name,
            "train_mse": float(mse_train),
            "train_r2": float(r2_train),
            "val_mse": float(mse_val) if mse_val is not None else None,
            "val_r2": float(r2_val) if r2_val is not None else None,
            "test_mse": float(mse_test) if mse_test is not None else None,
            "test_r2": float(r2_test) if r2_test is not None else None,
            "plots": plots
        })

    logger.info("✅ Завершено роботу run_multi_regression")
    return results

# ============================================================
# 🔹 Функція для запуску мульти-класифікації
# ============================================================
def run_multi_classification(X_train, Y_train, X_val=None, Y_val=None, X_test=None, Y_test=None):
    """
        Запускає MultiOutput класифікацію для кількох моделей (RandomForest, GradientBoosting, LogisticRegression).
        Обчислює метрики (Accuracy, F1) для train/val/test та будує графіки (confusion matrix, ROC, PR).
        """
    logger.info(">>> Старт функції run_multi_classification")
    results = []

    base_models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=200, random_state=42)
    }

    for name, base_model in base_models.items():
        print(f"\n=== Запуск моделі {name} (MultiOutput) ===")
        model = MultiOutputClassifier(base_model)
        model.fit(X_train, Y_train)

        # --- TRAIN ---
        preds_train = model.predict(X_train)
        Y_train_np = np.array(Y_train)
        preds_train_np = np.array(preds_train)

        accs, f1s = [], []
        confusion_plots_train = []

        for i in range(Y_train_np.shape[1]):
            accs.append(accuracy_score(Y_train_np[:, i], preds_train_np[:, i]))
            f1s.append(f1_score(Y_train_np[:, i], preds_train_np[:, i], average="macro"))

            cm = confusion_matrix(Y_train_np[:, i], preds_train_np[:, i])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Передбачені класи")
            ax.set_ylabel("Фактичні класи")
            ax.set_title(f"{name} — Target {i} (Train)")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_b64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            confusion_plots_train.append(plot_b64)

        acc_train = np.mean(accs)
        f1_train = np.mean(f1s)
        logger.info(f"[{name}] Train Accuracy={acc_train:.3f}, F1={f1_train:.3f}")

        # --- VALIDATION ---
        acc_val, f1_val, confusion_plots_val, roc_plots_val, pr_plots_val = None, None, [], [], []
        if X_val is not None and Y_val is not None:
            preds_val = model.predict(X_val)
            Y_val_np = np.array(Y_val)
            preds_val_np = np.array(preds_val)

            accs_val, f1s_val = [], []
            for i in range(Y_val_np.shape[1]):
                accs_val.append(accuracy_score(Y_val_np[:, i], preds_val_np[:, i]))
                f1s_val.append(f1_score(Y_val_np[:, i], preds_val_np[:, i], average="macro"))

                cm = confusion_matrix(Y_val_np[:, i], preds_val_np[:, i])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
                ax.set_title(f"{name} — Target {i} (Val)")
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                confusion_plots_val.append(base64.b64encode(buf.read()).decode("utf-8"))
                plt.close(fig)

                # ROC + PR curves (якщо є predict_proba)
                if hasattr(model.estimators_[i], "predict_proba"):
                    y_score = model.estimators_[i].predict_proba(X_val)
                    classes = np.unique(Y_val_np[:, i])
                    y_bin = label_binarize(Y_val_np[:, i], classes=classes)

                    # ROC
                    fig, ax = plt.subplots()
                    for j, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_bin[:, j], y_score[:, j])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f"Class {cls} (AUC={roc_auc:.2f})")
                    ax.plot([0,1],[0,1],"k--")
                    ax.legend()
                    ax.set_title(f"{name} ROC (Val Target {i})")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    roc_plots_val.append(base64.b64encode(buf.read()).decode("utf-8"))
                    plt.close(fig)

                    # Precision-Recall
                    fig, ax = plt.subplots()
                    for j, cls in enumerate(classes):
                        precision, recall, _ = precision_recall_curve(y_bin[:, j], y_score[:, j])
                        ax.plot(recall, precision, label=f"Class {cls}")
                    ax.legend()
                    ax.set_title(f"{name} PR Curve (Val Target {i})")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    pr_plots_val.append(base64.b64encode(buf.read()).decode("utf-8"))
                    plt.close(fig)

            acc_val = np.mean(accs_val)
            f1_val = np.mean(f1s_val)

        # --- TEST ---
        acc_test, f1_test, confusion_plots_test, roc_plots_test, pr_plots_test = None, None, [], [], []
        if X_test is not None and Y_test is not None:
            preds_test = model.predict(X_test)
            Y_test_np = np.array(Y_test)
            preds_test_np = np.array(preds_test)

            accs_test, f1s_test = [], []
            for i in range(Y_test_np.shape[1]):
                accs_test.append(accuracy_score(Y_test_np[:, i], preds_test_np[:, i]))
                f1s_test.append(f1_score(Y_test_np[:, i], preds_test_np[:, i], average="macro"))

                cm = confusion_matrix(Y_test_np[:, i], preds_test_np[:, i])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax)
                ax.set_title(f"{name} — Target {i} (Test)")
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                confusion_plots_test.append(base64.b64encode(buf.read()).decode("utf-8"))
                plt.close(fig)

                # ROC + PR curves (якщо є predict_proba)
                if hasattr(model.estimators_[i], "predict_proba"):
                    y_score = model.estimators_[i].predict_proba(X_test)
                    classes = np.unique(Y_test_np[:, i])
                    y_bin = label_binarize(Y_test_np[:, i], classes=classes)

                    # ROC
                    fig, ax = plt.subplots()
                    for j, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_bin[:, j], y_score[:, j])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f"Class {cls} (AUC={roc_auc:.2f})")
                    ax.plot([0,1],[0,1],"k--")
                    ax.legend()
                    ax.set_title(f"{name} ROC (Test Target {i})")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    roc_plots_test.append(base64.b64encode(buf.read()).decode("utf-8"))
                    plt.close(fig)

                    # Precision-Recall
                    fig, ax = plt.subplots()
                    for j, cls in enumerate(classes):
                        precision, recall, _ = precision_recall_curve(y_bin[:, j], y_score[:, j])
                        ax.plot(recall, precision, label=f"Class {cls}")
                    ax.legend()
                    ax.set_title(f"{name} PR Curve (Test Target {i})")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    pr_plots_test.append(base64.b64encode(buf.read()).decode("utf-8"))
                    plt.close(fig)

            acc_test = np.mean(accs_test)
            f1_test = np.mean(f1s_test)

        results.append({
            "model": name,
            "train_accuracy": float(acc_train),
            "train_f1": float(f1_train),
            "val_accuracy": float(acc_val) if acc_val is not None else None,
            "val_f1": float(f1_val) if f1_val is not None else None,
            "test_accuracy": float(acc_test) if acc_test is not None else None,
            "test_f1": float(f1_test) if f1_test is not None else None,
            "plots": {"confusion_plots_train": [
                {"title": f"Матриця неточностей (Train) — Target {i}", "plot": p}
                for i, p in enumerate(confusion_plots_train)
            ],
            "confusion_plots_val": [
                {"title": f"Матриця неточностей (Val) — Target {i}", "plot": p}
                for i, p in enumerate(confusion_plots_val)
            ],
            "confusion_plots_test": [
                {"title": f"Матриця неточностей (Test) — Target {i}", "plot": p}
                for i, p in enumerate(confusion_plots_test)
            ],
            "roc_plots_val": [
                {"title": f"ROC‑крива (Val) — Target {i}", "plot": p}
                for i, p in enumerate(roc_plots_val)
            ],
            "pr_plots_val": [
                {"title": f"Precision‑Recall крива (Val) — Target {i}", "plot": p}
                for i, p in enumerate(pr_plots_val)
            ],
            "roc_plots_test": [
                {"title": f"ROC‑крива (Test) — Target {i}", "plot": p}
                for i, p in enumerate(roc_plots_test)
            ],
            "pr_plots_test": [
                {"title": f"Precision‑Recall крива (Test) — Target {i}", "plot": p}
                for i, p in enumerate(pr_plots_test)
            ]}
        })

    logger.info("✅ Завершено роботу run_multi_classification")
    return results


# ============================================================
# 🔹 Ендпоінт для запуску регресійних моделей з RandomizedSearch
# ============================================================
@router.post("/run_regression_random")
async def run_regression_random_endpoint():
    """
    Запускає регресійні моделі (Logistic, RF, GB, XGBoost) з RandomizedSearchCV.
    """
    logger.info(">>> Виклик ендпоінта /run_regression_random")

    X_train, X_val, X_test = basic_state.get("x_train"), basic_state.get("x_val"), basic_state.get("x_test")
    Y_train, Y_val, Y_test = basic_state.get("y_train"), basic_state.get("y_val"), basic_state.get("y_test")

    if X_train is None or X_val is None or X_test is None or Y_train is None or Y_val is None or Y_test is None:
        logger.error("Train/Val дані відсутні")
        raise HTTPException(status_code=400, detail="Спочатку виконайте train/test/validation split")

    results = run_regression_models_random(X_train, X_val, X_test, Y_train, Y_val, Y_test, n_iter=30)
    log_transformation("Regression Models (RandomizedSearch)", None, "Logistic, RF, GB, XGBoost")
    logger.info("✅ Завершено запуск регресійних моделей")

    return {"top_results": results}


# ============================================================
# 🔹 Ендпоінт для запуску класифікаційних моделей з RandomizedSearch
# ============================================================
@router.post("/run_classification_random")
async def run_classification_random_endpoint():
    """
    Запускає класифікаційні моделі (Logistic, DecisionTree, RF, SVM) з RandomizedSearchCV.
    """
    logger.info(">>> Виклик ендпоінта /run_classification_random")

    X_train, X_val, X_test = basic_state.get("x_train"), basic_state.get("x_val"), basic_state.get("x_test")
    Y_train, Y_val, Y_test = basic_state.get("y_train"), basic_state.get("y_val"), basic_state.get("y_test")

    if X_train is None or X_val is None or X_test is None or Y_train is None or Y_val is None or Y_test is None:
        logger.error("Train/Val дані відсутні")
        raise HTTPException(status_code=400, detail="Спочатку виконайте train/test/validation split")

    results = run_classification_models_random(X_train, X_val, X_test, Y_train, Y_val, Y_test, n_iter=30)
    log_transformation("Classification Models (RandomizedSearch)", None, "Logistic, DecisionTree, RF, SVM")
    logger.info("✅ Завершено запуск класифікаційних моделей")

    return {"top_results": results}


# ============================================================
# 🔹 Ендпоінт для запуску кластеризаційних моделей
# ============================================================
@router.post("/run_clustering")
async def run_clustering_endpoint():
    """
    Запускає кластеризаційні моделі (KMeans, DBSCAN, Spectral, GMM).
    """
    logger.info(">>> Виклик ендпоінта /run_clustering")

    X_train, Y_train = basic_state.get("x_train"), basic_state.get("y_train")
    X_val, Y_val = basic_state.get("x_val"), basic_state.get("y_val")
    X_test, Y_test = basic_state.get("x_test"), basic_state.get("y_test")

    if X_train is None or Y_train is None:
        logger.error("Train дані відсутні")
        raise HTTPException(status_code=400, detail="Спочатку завантажте дані")

    results = run_clustering_models(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test)
    log_transformation("Clustering Models", None, "KMeans, DBSCAN, Spectral, GMM")
    logger.info("✅ Завершено запуск кластеризаційних моделей")

    return {"top_results": results,
            "validation_used": X_val is not None,
            "test_used": X_test is not None}


# ============================================================
# 🔹 Ендпоінт для запуску кластеризації без цільової змінної
# ============================================================
@router.post("/run_no_target_clustering")
async def run_no_target_clustering_endpoint():
    """
    Запускає кластеризацію без цільової змінної (unsupervised).
    """
    logger.info(">>> Виклик ендпоінта /run_no_target_clustering")

    df = basic_state.get("dataframe")
    X = basic_state.get("X")

    if df is None or X is None:
        logger.error("Дані відсутні у basic_state")
        raise HTTPException(status_code=400, detail="Спочатку завантажте дані")

    # Вибираємо колонки
    X = df[X]
    if isinstance(X, pd.Series):
        X = X.to_frame()

    logger.info(f"Вибрані колонки: {X.columns.tolist()}")
    logger.error(f"Перші рядки:\n{X.head()}")

    # 🔹 Перевірка типу даних
    if isinstance(X, pd.DataFrame):
        logger.debug(f"Отримано DataFrame з формою {X.shape}")
        X_num = X.select_dtypes(include=[np.number])
        if X_num.empty:
            raise HTTPException(status_code=400, detail="У DataFrame немає числових ознак для кластеризації")
        X_np = X_num.values
    elif isinstance(X, np.ndarray):
        logger.debug(f"Отримано ndarray з формою {X.shape}, dtype={X.dtype}")
        if not np.issubdtype(X.dtype, np.number):
            raise HTTPException(status_code=400, detail="ndarray містить нечислові значення")
        X_np = X
    else:
        logger.error(f"[ERROR] Неправильний тип даних: {type(X)}")
        raise HTTPException(status_code=400, detail="Очікується DataFrame або ndarray")

    # 🔹 Розділення на train/test
    X_train, X_test = train_test_split(X_np, test_size=0.2, random_state=42)
    logger.debug(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 🔹 Виклик кластеризації
    results = run_no_target_clustering(X_train, X_test)

    logger.info("Кластеризація завершена, повертаємо результати")

    return {
        "results": results,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "dtype": str(X_np.dtype),
        "original_type": str(type(X))
    }

# ============================================================
# 🔹 Ендпоінт для запуску MultiOutput регресії
# ============================================================
@router.post("/run_multi_regression")
async def run_multi_regression_endpoint():
    """
    Запускає MultiOutput регресійні моделі (RandomForest, GradientBoosting).
    """
    logger.info(">>> Виклик ендпоінта /run_multi_regression")
    X_train = basic_state.get("x_train")
    Y_train = basic_state.get("y_train")
    X_val = basic_state.get("x_val")
    Y_val = basic_state.get("y_val")
    X_test = basic_state.get("x_test")
    Y_test = basic_state.get("y_test")

    if X_train is None or Y_train is None:
        logger.error("Train дані відсутні")
        raise HTTPException(status_code=400, detail="Спочатку завантажте дані")

    results = run_multi_regression(
        X_train, Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test
    )

    # Логування у історію
    log_transformation("Multi Regression Models", None, "RandomForest, GradientBoosting")
    logger.info("✅ Завершено запуск MultiOutput регресії")

    return {"results": results}

# ============================================================
# 🔹 Ендпоінт для запуску MultiOutput класифікації
# ============================================================
@router.post("/run_multi_classification")
async def run_multi_classification_endpoint():
    """
    Запускає MultiOutput класифікаційні моделі (RandomForest, GradientBoosting, LogisticRegression).
    """
    logger.info(">>> Виклик ендпоінта /run_multi_classification")

    X_train = basic_state.get("x_train")
    Y_train = basic_state.get("y_train")
    X_val = basic_state.get("x_val")
    Y_val = basic_state.get("y_val")
    X_test = basic_state.get("x_test")
    Y_test = basic_state.get("y_test")

    if X_train is None or Y_train is None:
        logger.error("Train дані відсутні")
        raise HTTPException(status_code=400, detail="Спочатку завантажте дані")

    results = run_multi_classification(
        X_train, Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test
    )

    # Логування у історію
    log_transformation("Multi Classification Models", None, "RandomForest, GradientBoosting")
    logger.info("✅ Завершено запуск MultiOutput класифікації")

    return {"results": results}


# ============================================================
# 🔹 Ендпоінт для запуску змішаного прогнозу
# ============================================================
@router.post("/run_hybrid_prediction")
async def run_hybrid_prediction_endpoint(request: dict):
    logger.info(">>> Виклик ендпоінта /run_hybrid_prediction")
    logger.debug(f"Отримано request: {request}")

    X_train = basic_state.get("x_train")
    X_val = basic_state.get("x_val")
    y_train = basic_state.get("y_train")
    y_val = basic_state.get("y_val")

    numeric_targets = request.get("numeric_targets", [])
    categorical_targets = request.get("categorical_targets", [])

    logger.info(f"Цільові ознаки: numeric={numeric_targets}, categorical={categorical_targets}")

    results = {"classification": [], "regression": []}

    def plot_to_base64(fig):
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_b64

    # 🔹 Класифікація
    logger.info("Початок класифікації")
    classifiers = {
        "LogisticRegression": (LogisticRegression(max_iter=500), {"C":[0.1,1,10]}),
        "RandomForestClassifier": (RandomForestClassifier(), {"n_estimators":[50,100,200],"max_depth":[None,5,10]}),
        "GradientBoostingClassifier": (GradientBoostingClassifier(), {"n_estimators":[50,100],"learning_rate":[0.05,0.1]}),
        "XGBClassifier": (XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"), {"n_estimators":[50,100],"max_depth":[3,5]})
    }

    for target in categorical_targets:
        logger.debug(f"Класифікація для {target}")
        y_train_target = y_train[target]
        y_val_target = y_val[target]

        top_models = []
        for name,(model,param_grid) in classifiers.items():
            search = RandomizedSearchCV(model,param_grid,n_iter=3,cv=3,scoring="f1_weighted",random_state=42)
            search.fit(X_train,y_train_target)
            best_model = search.best_estimator_
            y_pred_val = best_model.predict(X_val)

            # Метрики
            acc = accuracy_score(y_val_target,y_pred_val)
            f1 = f1_score(y_val_target,y_pred_val,average="weighted")
            prec = precision_score(y_val_target,y_pred_val,average="weighted")
            rec = recall_score(y_val_target,y_pred_val,average="weighted")
            bal_acc = balanced_accuracy_score(y_val_target,y_pred_val)
            mcc = matthews_corrcoef(y_val_target,y_pred_val)
            kappa = cohen_kappa_score(y_val_target,y_pred_val)
            logloss = None
            if hasattr(best_model,"predict_proba"):
                y_prob = best_model.predict_proba(X_val)
                logloss = log_loss(y_val_target,y_prob)

            top_models.append({
                "model": name,
                "params": search.best_params_,
                "metrics": {
                    "accuracy":acc,"f1_score":f1,"precision":prec,"recall":rec,
                    "balanced_accuracy":bal_acc,"mcc":mcc,"kappa":kappa,"logloss":logloss
                },
                "estimator": best_model
            })

        # топ‑3 за F1
        top_models = sorted(top_models,key=lambda x:x["metrics"]["f1_score"],reverse=True)[:3]

        # Додаємо графіки для кожної з топ‑3 моделей
        for m in top_models:
            est = m["estimator"]
            y_pred_val = est.predict(X_val)

            # Confusion matrix
            cm = confusion_matrix(y_val_target, y_pred_val)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix ({m['model']})")
            m["confusion_matrix_plot"] = plot_to_base64(fig)

            # ROC curve
            if hasattr(est, "predict_proba"):
                y_score = est.predict_proba(X_val)
                classes = np.unique(y_val_target)
                y_bin = label_binarize(y_val_target, classes=classes)
                fig, ax = plt.subplots()
                for i, cls in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_bin[:,i], y_score[:,i])
                    roc_auc = auc(fpr,tpr)
                    ax.plot(fpr,tpr,label=f"Class {cls} (AUC={roc_auc:.2f})")
                ax.plot([0,1],[0,1],"k--")
                ax.legend()
                ax.set_title(f"ROC Curve ({m['model']})")
                m["roc_plot"] = plot_to_base64(fig)

                # Precision-Recall curve
                fig, ax = plt.subplots()
                for i, cls in enumerate(classes):
                    precision, recall, _ = precision_recall_curve(y_bin[:,i], y_score[:,i])
                    ax.plot(recall, precision, label=f"Class {cls}")
                ax.set_title(f"Precision-Recall Curve ({m['model']})")
                ax.legend()
                m["pr_curve"] = plot_to_base64(fig)

            # Feature importance (для дерев’яних моделей)
            if hasattr(est, "feature_importances_"):
                fig, ax = plt.subplots()
                importances = est.feature_importances_
                indices = np.argsort(importances)[::-1]
                ax.bar(range(len(importances)), importances[indices])
                ax.set_title(f"Feature Importance ({m['model']})")
                m["feature_importance_plot"] = plot_to_base64(fig)

        results["classification"].append({
            "target": target,
            "top_models": [{k:v for k,v in m.items() if k!="estimator"} for m in top_models]
        })
    logger.info("Класифікація завершена")

    # 🔹 Регресія
    logger.info("Початок регресії")
    regressors = {
        "LinearRegression": (LinearRegression(), {}),
        "RandomForestRegressor": (RandomForestRegressor(), {"n_estimators":[50,100,200],"max_depth":[None,5,10]}),
        "GradientBoostingRegressor": (GradientBoostingRegressor(), {"n_estimators":[50,100],"learning_rate":[0.05,0.1]}),
        "XGBRegressor": (XGBRegressor(), {"n_estimators":[50,100],"max_depth":[3,5]})
    }

    for target in numeric_targets:
        print(f"[INFO] Регресія для {target}")
        y_train_target = y_train[target]
        y_val_target = y_val[target]

        top_models = []
        for name,(model,param_grid) in regressors.items():
            search = RandomizedSearchCV(model,param_grid,n_iter=3,cv=3,scoring="r2",random_state=42)
            search.fit(X_train,y_train_target)
            best_model = search.best_estimator_
            y_pred_val = best_model.predict(X_val)

            # Метрики
            r2 = r2_score(y_val_target,y_pred_val)
            mae = mean_absolute_error(y_val_target,y_pred_val)
            rmse = np.sqrt(mean_squared_error(y_val_target,y_pred_val))
            median_ae = median_absolute_error(y_val_target,y_pred_val)
            msle = mean_squared_log_error(np.maximum(y_val_target,0)+1, np.maximum(y_pred_val,0)+1)
            explained_var = explained_variance_score(y_val_target,y_pred_val)
            adj_r2 = 1 - (1-r2)*(len(y_val_target)-1)/(len(y_val_target)-X_val.shape[1]-1)

            top_models.append({
                "model": name,
                "params": search.best_params_,
                "metrics": {
                    "r2_score":r2,"adjusted_r2":adj_r2,"mae":mae,"rmse":rmse,
                    "median_ae":median_ae,"msle":msle,"explained_variance":explained_var
                },
                "estimator": best_model,
                "pred_val": y_pred_val
            })

        # топ‑3 за R²
        top_models = sorted(top_models,key=lambda x:x["metrics"]["r2_score"],reverse=True)[:3]

        # Додаємо графіки для кожної з топ‑3 моделей
        for m in top_models:
            y_pred_val = m["pred_val"]

            # Scatter plot
            fig, ax = plt.subplots()
            ax.scatter(y_val_target, y_pred_val, alpha=0.6)
            ax.set_xlabel("True values")
            ax.set_ylabel("Predicted values")
            ax.set_title(f"Scatter ({m['model']})")
            m["scatter_plot"] = plot_to_base64(fig)

            # Residual plot
            residuals = y_val_target - y_pred_val
            fig, ax = plt.subplots()
            ax.scatter(y_pred_val, residuals, alpha=0.6, color="red")
            ax.axhline(y=0, color="black", linestyle="--")
            ax.set_xlabel("Predicted values")
            ax.set_ylabel("Residuals")
            ax.set_title(f"Residuals ({m['model']})")
            m["residual_plot"] = plot_to_base64(fig)

            # Distribution of residuals
            fig, ax = plt.subplots()
            sns.histplot(residuals, bins=20, kde=True, ax=ax)
            ax.set_title(f"Residuals Distribution ({m['model']})")
            m["residuals_distribution"] = plot_to_base64(fig)

            # Q-Q plot
            fig, ax = plt.subplots()
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot ({m['model']})")
            m["qq_plot"] = plot_to_base64(fig)

            # Feature importance (для дерев’яних моделей)
            if hasattr(m["estimator"], "feature_importances_"):
                fig, ax = plt.subplots()
                importances = m["estimator"].feature_importances_
                indices = np.argsort(importances)[::-1]
                ax.bar(range(len(importances)), importances[indices])
                ax.set_title(f"Feature Importance ({m['model']})")
                m["feature_importance_plot"] = plot_to_base64(fig)

        results["regression"].append({
            "target": target,
            "top_models": [{k: v for k, v in m.items() if k not in ["estimator", "pred_val"]} for m in top_models]
        })
    logger.info("Регресія завершена")

    logger.info("✅ Змішаний прогноз завершено")
    return results


# ============================================================
# 🔹 Ендпоінт для отримання цільових ознак для змішаного прогнозу
# ============================================================
@router.get("/get_target_features_for_hybrid")
async def get_target_features():
    logger.info(">>> Виклик ендпоінта /get_target_features_for_hybrid")

    df = basic_state.get("dataframe")
    Y = basic_state.get("Y")

    if df is None or Y is None:
        logger.error("Дані або цільові ознаки відсутні")
        raise HTTPException(status_code=400, detail="Дані або цільові ознаки відсутні")

    if isinstance(Y, list):
        target_features = Y
    elif isinstance(Y, str):
        target_features = [Y]
    else:
        logger.error("Неправильний формат Y")
        raise HTTPException(status_code=400, detail="Неправильний формат Y")

    logger.info(f"✅ Цільові ознаки: {target_features}")
    return {"target_features": target_features}
