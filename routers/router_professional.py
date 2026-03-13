import io, base64, datetime, pickle

import numpy as np
import pandas as pd
import sklearn
from fastapi import APIRouter, Request, UploadFile, File, Body, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from TEST_PROF_ALGRS.start_task_detector import run_prediction_task
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from logging_config import logger

router = APIRouter(prefix="/professional_mode", tags=["Professional mode"])

templates = Jinja2Templates(directory="templates")

basic_state = {
    "dataframe": None,
    "model": None,
    "task_type": None,
    "target_column": None,
    "feature_columns": None
}

# глобальний словник для збереження інформації про датасет
dataset_info = {}

@router.get("/", response_class=HTMLResponse)
async def professional_home_page(request: Request):
    return templates.TemplateResponse(
        "professional/professional_mode_index.html",
        {"request": request}
    )


@router.post("/upload", response_class=HTMLResponse)
async def upload_csv(
    request: Request,
    file: UploadFile = File(...)
):
    # 1. Перевірка формату файлу
    if not file.filename.endswith(".csv"):
        logger.error("Крок 1: Файл не є CSV")
        raise HTTPException(status_code=400, detail="Потрібен CSV файл")
    logger.info("Крок 1: Файл має правильне розширення .csv")

    # 2. Читання вмісту файлу
    contents = await file.read()
    logger.debug("Крок 2: Вміст файлу успішно прочитано")

    # 3. Завантаження у DataFrame
    df = pd.read_csv(io.BytesIO(contents))
    logger.info("Крок 3: Файл перетворено у DataFrame")

    # 4. Збереження у глобальний стан
    basic_state["dataframe"] = df
    logger.info("Крок 4: DataFrame збережено у пам'ять (basic_state)")

    # 5. Формування прев’ю
    preview = df.head(10)
    logger.debug("Крок 5: Сформовано прев’ю з перших 10 рядків")

    # 6. Отримання колонок та типів даних
    columns = df.columns.tolist()
    dtypes = {col: str(df[col].dtype) for col in columns}
    logger.info("Крок 6: Отримано список колонок та їх типи даних")

    # Зберігаємо у глобальний словник
    global dataset_info
    dataset_info = {
        "dataset_name": file.filename,
        "columns": columns,
        "dtypes": dtypes
    }

    # 7. Повернення результату у шаблон
    logger.info("Крок 7: Повертаємо результат у шаблон professional_mode_index.html")

    return templates.TemplateResponse(
        "professional/professional_mode_index.html",
        {
            "request": request,
            "table": preview.to_dict(orient="records"),
            "columns": columns,
            "dtypes": dtypes
        }
    )


def detect_task_type(df: pd.DataFrame, target: str = None, file_path: str = None) -> str:
    logger.info("Завдання визначено у detect_task_type")

    # Якщо df не передано → читаємо з файлу
    if df is None:
        if not file_path or not os.path.exists(file_path):
            logger.error(f"Файл '{file_path}' не знайдено.")
            raise FileNotFoundError(f"❌ Файл '{file_path}' не знайдено.")
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Датасет успішно завантажено з файлу {file_path}. Розмір: {df.shape}")
        except Exception as e:
            logger.error(f"Помилка при читанні файлу: {e}")
            raise ValueError(f"❌ Помилка при читанні файлу: {e}")
    else:
        logger.info(f"✅ Датасет успішно завантажено. Розмір: {df.shape}")

    # === Визначення типу задачі ===
    if target is None:
        logger.warning("Цільова змінна не задана → задача кластеризації.")
        return "clustering"

    if target not in df.columns:
        logger.error(f"Колонка '{target}' відсутня у датасеті. Доступні колонки: {list(df.columns)}")
        raise ValueError(f"❌ Колонка '{target}' відсутня у датасеті. Доступні колонки: {list(df.columns)}")

    target_dtype = df[target].dtype
    unique_values = df[target].nunique()

    logger.debug(f"Тип даних цільової змінної '{target}': {target_dtype}, унікальних значень: {unique_values}")

    if pd.api.types.is_numeric_dtype(df[target]):
        if unique_values <= 10:
            logger.info("Визначено задачу класифікації (числова змінна з малим числом унікальних значень).")
            return "classification"
        else:
            logger.info("Визначено задачу регресії.")
            return "regression"
    else:
        logger.info("Визначено задачу класифікації.")
        return "classification"


@router.post("/detect_task")
async def detect_task(payload: dict = Body(...)):
    """
    Ендпоінт для визначення типу задачі машинного навчання.

    Приймає JSON-повідомлення з цільовою колонкою.
    Використовує поточний DataFrame із глобального стану (basic_state).
    Викликає функцію detect_task_type для визначення типу задачі:
    - кластеризація, якщо цільова змінна не задана
    - класифікація або регресія залежно від типу даних та кількості унікальних значень
    Повертає JSON з назвою цільової колонки та визначеним типом задачі.
    """

    # Отримуємо DataFrame із глобального стану
    df = basic_state.get("dataframe")
    target_column = payload.get("target")

    # Перевірка: чи є дані для аналізу
    if df is None:
        logger.error("Немає даних для аналізу")
        raise HTTPException(status_code=400, detail="Немає даних для аналізу")

    # Перевірка: чи задано цільову колонку
    if not target_column:
        logger.error("Не задано цільову колонку")
        raise HTTPException(status_code=400, detail="Не задано цільову колонку")

    try:
        # Виклик функції визначення типу задачі
        task_type = detect_task_type(df=df, target=target_column)
        logger.info(f"Задача визначена: {task_type} для цільової змінної '{target_column}'")
    except Exception as e:
        logger.critical(f"Помилка при визначенні задачі: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "target_column": target_column,
        "task_type": task_type
    }


def normalize_params(params: dict) -> dict:
    """
    Функція нормалізації параметрів.

    Приймає словник параметрів, де значення можуть бути рядками.
    Виконує перетворення:
    - "true"/"false" → булеві значення
    - рядки з числами → int або float
    - інші рядки залишаються без змін
    Повертає новий словник із нормалізованими значеннями.
    """

    normalized = {}
    logger.info("Початок нормалізації параметрів")

    for key, value in params.items():
        logger.debug(f"Обробка ключа '{key}' зі значенням '{value}' ({type(value)})")

        if isinstance(value, str):
            # Перетворення рядків "true"/"false" у булеві значення
            if value.lower() in ["true", "false"]:
                normalized[key] = value.lower() == "true"
                logger.debug(f"Ключ '{key}' нормалізовано як булеве: {normalized[key]}")
            else:
                try:
                    # Перетворення рядків у числа (float або int)
                    if "." in value:
                        normalized[key] = float(value)
                        logger.debug(f"Ключ '{key}' нормалізовано як float: {normalized[key]}")
                    else:
                        normalized[key] = int(value)
                        logger.debug(f"Ключ '{key}' нормалізовано як int: {normalized[key]}")
                except ValueError:
                    # Якщо перетворення не вдалося — залишаємо рядок
                    normalized[key] = value
                    logger.warning(f"Ключ '{key}' залишено як рядок: {normalized[key]}")
        else:
            # Якщо значення вже не рядок — залишаємо як є
            normalized[key] = value
            logger.debug(f"Ключ '{key}' залишено без змін: {normalized[key]}")

    logger.info("Нормалізація параметрів завершена")
    return normalized


@router.post("/train")
async def train_model(payload: dict = Body(...)):
    """
    Ендпоінт для навчання моделі машинного навчання.

    Приймає JSON-повідомлення з параметрами:
    - target_column: цільова змінна
    - model_name: назва моделі
    - params: параметри моделі
    - task_type: тип задачі (класифікація, регресія, кластеризація)

    Виконує:
    1. Отримання payload та нормалізацію параметрів
    2. Запуск функції run_prediction_task
    3. Збереження стану у глобальному словнику basic_state
    4. Формування пакету моделі
    5. Збереження моделі у файл
    6. Формування відповіді з усією інформацією про модель
    """
    logger.info("================================")
    logger.info("=== Початок ендпоінта /train ===")
    logger.info("================================")

    # Етап 1: Отримуємо payload
    logger.info(">>> Етап 1: Отримуємо payload")
    target_column = payload.get("target")
    model_name = payload.get("model_name")
    params = normalize_params(payload.get("params", {}))
    task_type = payload.get("task_type")

    logger.debug(f"target_column: {target_column}")
    logger.debug(f"model_name: {model_name}")
    logger.debug(f"params: {params}")
    logger.debug(f"task_type: {task_type}")
    logger.debug(f"Типи даних у params: { {k: type(v) for k, v in params.items()} }")

    df = basic_state.get("dataframe")
    if df is None:
        logger.error("❌ У basic_state немає dataframe")
        raise HTTPException(status_code=400, detail="Немає даних для навчання")
    else:
        logger.info(f"Розмір датасету: {df.shape}")

    # Етап 2: Запускаємо run_prediction_task
    logger.info(">>> Етап 2: Запускаємо run_prediction_task")
    try:
        result = run_prediction_task(
            df=df,
            target=target_column if target_column else None,
            model_name=model_name,
            params=params,
            task_type=task_type
        )
    except Exception as e:
        logger.critical(f"Помилка при запуску run_prediction_task: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    logger.debug(f"result keys: {list(result.keys())}")

    # Етап 3: Збереження стану
    logger.info(">>> Етап 3: Збереження стану")
    basic_state["model"] = result.get("model") or result.get("pipeline")
    basic_state["task_type"] = result.get("task_type")
    basic_state["target_column"] = target_column if target_column else None
    basic_state["model_name"] = model_name
    basic_state["params"] = params
    basic_state["feature_columns"] = result.get("features")
    logger.debug(f"basic_state: {basic_state}")

    # Етап 4: Формуємо пакет моделі
    logger.info(">>> Етап 4: Формуємо пакет моделі")
    if result.get("task_type") == "clustering":
        model_package = {
            "model": result.get("model"),
            "metadata": {
                "features": list(result.get("X_train").columns) if result.get("X_train") is not None else [],
                "metrics": result.get("metrics", {}),
                "params": params,
                "task_type": "clustering",
                "target_column": target_column,
                "plots": result.get("plots", []),
                "input_template": result.get("input_template")
            }
        }
    else:
        model_package = {
            "pipeline": result.get("pipeline"),
            "metadata": {
                "features": result.get("features"),
                "metrics": result.get("metrics", {}),
                "params": params,
                "task_type": result.get("task_type"),
                "target_column": target_column,
                "plots": result.get("plots", []),
                "input_template": result.get("input_template")
            }
        }
    # Логування всього словника model_package
    logger.debug("=== Вміст model_package ===")
    for key, value in model_package.items():
        # Якщо значення велике (наприклад, графіки у base64), обрізаємо для читабельності
        if isinstance(value, dict) and key == "metadata":
            logger.debug(f"{key}:")
            for meta_key, meta_val in value.items():
                if meta_key == "plots" and isinstance(meta_val, dict):
                    logger.debug(f"  {meta_key}: {list(meta_val.keys())} (base64 обрізано)")
                else:
                    logger.debug(f"  {meta_key}: {meta_val}")
        else:
            logger.debug(f"{key}: {value}")

    # Етап 5: Зберігаємо у файл
    logger.info(">>> Етап 5: Зберігаємо у файл")
    filepath = os.path.join("saved_models", f"{model_name}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model_package, f)
    logger.info(f"Модель збережено у файл: {filepath}")

    # Етап 6: Формуємо відповідь
    logger.info(">>> Етап 6: Формуємо відповідь")
    full_info = {
        "message": "✅ Модель успішно навчена",
        "task_type": result.get("task_type"),
        "target_column": target_column,
        "model_name": model_name,
        "params": params,
        "plots": result.get("plots", []),
        "features": list(result.get("X_train").columns) if result.get("task_type") == "clustering" and result.get(
            "X_train") is not None else result.get("features"),
        "metrics": result.get("metrics", {}),
        "input_template": result.get("input_template")
    }

    # Логування графіків
    plots = result.get("plots", [])
    if plots:
        logger.debug("=== Графіки (base64) ===")
        for name, img in plots.items():
            logger.debug(f"{name}: {img[:100]}...")

    # Логування повної інформації
    logger.info("=== Повна інформація про модель ===")
    for key, value in full_info.items():
        if key != 'plots':
            logger.info(f"{key}: {value}")

    logger.info("===============================")
    logger.info("=== Кінець ендпоінта /train ===")
    logger.info("================================")

    return full_info


@router.get("/download_model/{filename}")
async def download_model(filename: str):
    """
    Ендпоінт для завантаження збереженої моделі.

    Перевіряє наявність файлу у папці saved_models.
    Якщо файл існує — повертає його як FileResponse.
    Якщо ні — повертає помилку 404 із переліком доступних файлів.
    """
    folder = "saved_models"
    available_files = os.listdir(folder)
    logger.info(">>> Доступні моделі у saved_models")
    logger.debug(f"Список файлів: {available_files}")

    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        logger.error(f"Файл {filename} не знайдено. Доступні файли: {available_files}")
        raise HTTPException(status_code=404, detail=f"Файл {filename} не знайдено. Доступні файли: {available_files}")

    logger.info(f"Файл {filename} знайдено, відправляємо користувачу")
    return FileResponse(file_path, filename=filename)


@router.get("/compare_models", response_class=HTMLResponse)
async def compare_models_page(request: Request):
    """
    Ендпоінт для відображення сторінки порівняння моделей.
    """
    logger.info("Відкрито сторінку порівняння моделей")
    return templates.TemplateResponse(
        "professional/compare_model.html",
        {"request": request}
    )


@router.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """
    Ендпоінт для завантаження моделі користувачем.

    Зберігає файл у папку uploaded_models.
    Пробує розпакувати pickle-файл.
    Якщо успішно — повертає метадані моделі.
    Якщо ні — повертає помилку 400.
    """
    folder = "uploaded_models"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)

    logger.info(f"Отримано файл {file.filename}, зберігаємо у {folder}")
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"Файл {file.filename} успішно збережено")
    except Exception as e:
        logger.error(f"Помилка при збереженні файлу {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Не вдалося зберегти файл: {e}")

    try:
        with open(file_path, "rb") as f:
            model_package = pickle.load(f)
        logger.info(f"Файл {file.filename} успішно розпаковано")
    except Exception as e:
        logger.error(f"Не вдалося розпакувати файл {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Не вдалося розпакувати файл: {e}")

    metadata = model_package.get("metadata", {})
    logger.debug("Метадані моделі:")
    for key, value in metadata.items():
        if key == "plots" and isinstance(value, dict):
            # Для графіків виводимо тільки назви та перші 10 символів base64
            for plot_name, img in value.items():
                logger.debug(f"{key} -> {plot_name}: {img[:10]}...")
        else:
            logger.debug(f"{key}: {value}")

    return {"filename": file.filename, "metadata": metadata}



@router.post("/predict")
async def predict(payload: dict = Body(...)):
    """
    Ендпоінт для прогнозування нових даних за допомогою збереженої моделі (.pkl).

    Приймає JSON-повідомлення з:
    - model_name: назва моделі
    - input_data: словник із новими даними для прогнозу

    Виконує:
    1. Отримання payload
    2. Завантаження моделі з файлу
    3. Витяг метаданих
    4. Формування DataFrame з нових даних
    5. Виконання прогнозу
    6. Формування повної відповіді з усією інформацією
    """
    logger.info("==================================")
    logger.info("=== Початок ендпоінта /predict ===")
    logger.info("==================================")

    # Етап 1: Отримуємо payload
    logger.info(">>> Етап 1: Отримуємо payload")
    model_name = payload.get("model_name")
    input_data = payload.get("input_data")
    logger.debug(f"model_name: {model_name}")
    logger.debug(f"input_data: {input_data}")
    logger.debug(f"Типи даних у input_data: { {k: type(v) for k, v in input_data.items()} }")

    if not model_name or not input_data:
        logger.error("Необхідні поля: model_name, input_data")
        raise HTTPException(status_code=400, detail="❌ Необхідні поля: model_name, input_data")

    # Етап 2: Дістаємо шлях до моделі
    logger.info(">>> Етап 2: Дістаємо шлях до моделі")
    filepath = os.path.join("saved_models", f"{model_name}.pkl")
    logger.debug(f"filepath: {filepath}")

    if not os.path.exists(filepath):
        logger.error(f"Файл {filepath} не знайдено")
        raise HTTPException(status_code=404, detail=f"❌ Файл {filepath} не знайдено")

    try:
        # Етап 3: Завантажуємо модель
        logger.info(">>> Етап 3: Завантажуємо модель")
        with open(filepath, "rb") as f:
            model_package = pickle.load(f)
        logger.debug(f"model_package keys: {list(model_package.keys())}")

        # Етап 4: Витягуємо модель та метадані
        logger.info(">>> Етап 4: Витягуємо модель та метадані")
        model = model_package.get("model") or model_package.get("pipeline")
        metadata = model_package.get("metadata", {})
        logger.info(f"Тип моделі: {type(model).__name__}")
        for key, value in metadata.items():
            if key != "plots":
                logger.debug(f"{key}: {value}")

        features = metadata.get("features", [])
        task_type = metadata.get("task_type")
        target_column = metadata.get("target_column")
        metrics = metadata.get("metrics", {})
        params = metadata.get("params", {})
        input_template = metadata.get("input_template")

        # Виводимо графіки
        plots = metadata.get("plots", {})
        if plots:
            logger.debug(">>> Графіки (base64)")
            for name, img in plots.items():
                logger.debug(f"{name}: {img[:100]}...")

        # Етап 5: Формуємо DataFrame з нових даних
        logger.info(">>> Етап 5: Формуємо DataFrame з нових даних")
        try:
            input_data = {k: float(v) for k, v in input_data.items()}
        except ValueError as e:
            logger.error(f"Неможливо конвертувати дані у числа: {e}")
            raise HTTPException(status_code=400, detail=f"❌ Неможливо конвертувати дані у числа: {e}")

        X_new = pd.DataFrame([input_data], columns=features)
        logger.debug(f"X_new: {X_new}")
        logger.debug(f"Типи даних у X_new: {X_new.dtypes}")

        # Етап 6: Прогноз
        logger.info(">>> Етап 6: Виконуємо прогноз")
        try:
            prediction = model.predict(X_new)[0]
            logger.info(f"Результат predict: {prediction}")
        except AttributeError:
            prediction = model.fit_predict(X_new)[0]
            logger.info(f"Результат fit_predict: {prediction}")

        if isinstance(prediction, (np.generic, np.int64)):
            prediction = int(prediction)
        elif isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        logger.debug(f"Конвертований prediction: {prediction}")

        # Етап 7: Пайплайн
        logger.info(">>> Етап 7: Пайплайн")
        pipeline_steps = None
        if hasattr(model, "steps"):
            pipeline_steps = [str(step) for step in model.steps]
        logger.debug(f"pipeline_steps: {pipeline_steps}")

        # Етап 8: Повна інформація
        logger.info(">>> Етап 8: Формуємо повну інформацію про модель")
        full_info = {
            "model_name": model_name,
            "task_type": task_type,
            "model_type": type(model).__name__,
            "features": features,
            "input_data": input_data,
            "prediction": prediction,
            "pipeline_steps": pipeline_steps,
            "metrics": metrics,
            "params": params,
            "target_column": target_column,
            "input_template": input_template,
            "plots": plots,
            "library_version": {
                "sklearn": sklearn.__version__,
                "pandas": pd.__version__
            },
            "execution_time": str(pd.Timestamp.now())
        }

        for key, value in full_info.items():
            if key == "plots":
                for name, img in value.items():
                    logger.debug(f"{name}: {img[:100]}...")
            else:
                logger.info(f"{key}: {value}")

        logger.info("=================================")
        logger.info("=== Кінець ендпоінта /predict ===")
        logger.info("=================================")

        return full_info

    except Exception as e:
        logger.critical(f"Помилка при прогнозуванні: {e}")
        raise HTTPException(status_code=500, detail=f"❌ Помилка при прогнозуванні: {e}")


def safe_draw_string(c, x, y, text, font="DejaVuSans", size=12):
    """
    Допоміжна функція для малювання рядка у PDF.
    Якщо не вистачає місця на сторінці — створюється нова сторінка.
    """
    if y < 3*cm:
        c.showPage()
        c.setFont(font, size)
        y = A4[1] - 3*cm
    c.setFont(font, size)
    c.drawString(x, y, text)
    return y - 1*cm


@router.post("/generate_report")
async def generate_report(payload: dict = Body(...)):
    """
    Ендпоінт для генерації PDF-звіту про модель та її результати.

    Виконує:
    1. Реєстрацію шрифту
    2. Формування титульної сторінки
    3. Додавання інформації про датасет
    4. Основну інформацію про модель
    5. Ознаки, введені дані, прогноз
    6. Метрики та параметри
    7. Графіки, кроки пайплайну, шаблон вводу
    8. Версії бібліотек та час виконання
    """

    global dataset_info
    logger.info("Отримано дані для звіту")
    for key, value in payload.items():
        if key == "plots" and isinstance(value, dict):
            # Для графіків виводимо тільки ключі та перші 100 символів base64
            for plot_name, img in value.items():
                logger.debug(f"{key} -> {plot_name}: {img[:100]}...")
        else:
            logger.debug(f"{key}: {value}")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    # Реєстрація шрифту
    logger.info("➡️ Реєструємо шрифт...")
    font_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fonts", "DejaVuSans.ttf"))
    if os.path.isfile(font_path):
        pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", font_path))
        logger.info("✅ Шрифт DejaVuSans зареєстровано")
    else:
        logger.warning("❌ Шрифт не знайдено")

    # Титульна сторінка
    logger.info("➡️ Формуємо титульну сторінку...")
    c.setFont("DejaVuSans", 18)
    c.drawCentredString(A4[0]/2, A4[1]-3*cm, "ОФІЦІЙНИЙ ЗВІТ ПРО ПРОГНОЗУВАННЯ")
    c.setFont("DejaVuSans", 12)
    c.drawCentredString(A4[0]/2, A4[1]-4*cm, "Data Science Prediction System")
    c.drawCentredString(A4[0]/2, A4[1]-5*cm, f"Дата формування: {datetime.date.today()}")

    # Інформація про датасет
    if dataset_info:
        logger.info("➡️ Додаємо інформацію про датасет...")
        c.showPage()
        c.setFont("DejaVuSans-Bold", 14)
        c.drawString(2*cm, A4[1]-2*cm, "Інформація про датасет:")
        c.setFont("DejaVuSans", 12)
        c.drawString(2*cm, A4[1]-3*cm, f"Назва файлу: {dataset_info.get('dataset_name','N/A')}")
        y = A4[1]-5*cm
        for col, dtype in dataset_info.get("dtypes", {}).items():
            y = safe_draw_string(c, 2*cm, y, f"{col}: {dtype}")
        c.showPage()

    # Основна інформація
    logger.info("➡️ Додаємо основну інформацію...")
    model_name = payload.get("model_name","N/A")
    task_type = payload.get("task_type","N/A")
    model_type = payload.get("model_type","N/A")
    target_column = payload.get("target_column","N/A")
    c.setFont("DejaVuSans-Bold", 14)
    c.drawString(2*cm, A4[1]-2*cm, "Основна інформація:")
    c.setFont("DejaVuSans", 12)
    c.drawString(2*cm, A4[1]-3*cm, f"Модель: {model_name}")
    c.drawString(2*cm, A4[1]-4*cm, f"Тип задачі: {task_type}")
    c.drawString(2*cm, A4[1]-5*cm, f"Тип моделі: {model_type}")
    c.drawString(2*cm, A4[1]-6*cm, f"Цільова змінна: {target_column}")

    # Ознаки
    logger.info("➡️ Додаємо ознаки...")
    features = payload.get("features", [])
    c.setFont("DejaVuSans-Bold", 14)
    c.drawString(2*cm, A4[1]-8*cm, "Ознаки:")
    c.setFont("DejaVuSans", 12)
    y = A4[1]-9*cm
    for f in features:
        y = safe_draw_string(c, 2*cm, y, f)

    # Введені дані
    logger.info("➡️ Додаємо введені дані...")
    input_data = payload.get("input_data", {})
    c.setFont("DejaVuSans-Bold", 14)
    c.drawString(2*cm, y-2*cm, "Введені дані користувачем:")
    c.setFont("DejaVuSans", 12)
    y -= 3*cm
    for k,v in input_data.items():
        y = safe_draw_string(c, 2*cm, y, f"{k}: {v}")

    # Прогноз
    logger.info("➡️ Додаємо прогноз...")
    prediction = payload.get("prediction","")
    c.setFont("DejaVuSans-Bold", 14)
    c.drawString(2*cm, y-2*cm, "Передбачені значення:")
    c.setFont("DejaVuSans", 12)
    c.drawString(2*cm, y-3*cm, str(prediction))
    c.showPage()

    # Метрики
    logger.info("➡️ Додаємо метрики...")
    metrics = payload.get("metrics", {})
    c.setFont("DejaVuSans-Bold", 14)
    c.drawString(2*cm, A4[1]-2*cm, "Метрики моделі:")
    c.setFont("DejaVuSans", 12)
    y = A4[1]-3*cm
    for metric,val in metrics.items():
        y = safe_draw_string(c, 2*cm, y, f"{metric}: {val}")

    # Параметри
    logger.info("➡️ Додаємо параметри...")
    params = payload.get("params", {})
    c.showPage()
    c.setFont("DejaVuSans-Bold", 14)
    c.drawString(2*cm, A4[1]-2*cm, "Параметри моделі:")
    c.setFont("DejaVuSans", 12)
    y = A4[1]-3*cm
    for p,val in params.items():
        y = safe_draw_string(c, 2*cm, y, f"{p}: {val}")

    # Графіки
    plots = payload.get("plots", {})
    if plots:
        logger.info("➡️ Додаємо графіки...")
        c.showPage()
        c.setFont("DejaVuSans-Bold", 14)
        c.drawString(2 * cm, A4[1] - 2 * cm, "Графіки:")
        y = A4[1] - 4 * cm
        for plotName, base64img in plots.items():
            # Логування назви графіка та перших 100 символів base64
            logger.debug(f" Графік: {plotName}, дані: {base64img[:100]}...")

            imgdata = base64.b64decode(base64img)
            imgfile = io.BytesIO(imgdata)
            image = ImageReader(imgfile)
            img_width, img_height = image.getSize()
            max_width, max_height = 12 * cm, 8 * cm
            scale = min(max_width / img_width, max_height / img_height)
            new_width, new_height = img_width * scale, img_height * scale

            # Перевірка наявності місця на сторінці
            if y - new_height < 3 * cm:
                c.showPage()
                y = A4[1] - 4 * cm

            # Малюємо графік
            c.drawImage(image, 2 * cm, y - new_height, width=new_width, height=new_height)
            c.setFont("DejaVuSans", 12)
            c.drawString(2 * cm, y - new_height - 1 * cm, plotName)
            y -= (new_height + 3 * cm)

    # Кроки пайплайну
    steps = payload.get("pipeline_steps")
    if steps:
        logger.info("➡️ Додаємо кроки пайплайну...")
        c.showPage()
        c.setFont("DejaVuSans-Bold", 14)
        c.drawString(2 * cm, A4[1] - 2 * cm, "Кроки пайплайну:")
        c.setFont("DejaVuSans", 12)
        y = A4[1] - 3 * cm
        for s in steps:
            y = safe_draw_string(c, 2 * cm, y, str(s))
    else:
        logger.info("ℹ️ Кроки пайплайну відсутні")

    # Шаблон вводу
    logger.info("➡️ Додаємо шаблон вводу...")
    input_template = payload.get("input_template", {})
    c.showPage()
    c.setFont("DejaVuSans-Bold", 14)
    c.drawString(2 * cm, A4[1] - 2 * cm, "Шаблон вводу:")
    c.setFont("DejaVuSans", 12)
    y = A4[1] - 3 * cm
    for feature, placeholder in input_template.items():
        logger.debug(f"{feature}: {placeholder}")
        y = safe_draw_string(c, 2 * cm, y, f"{feature}: {placeholder}")


    # Версії бібліотек
    logger.info("➡️ Додаємо версії бібліотек...")
    libs = payload.get("library_version", {})
    c.showPage()
    c.setFont("DejaVuSans-Bold", 14)
    c.drawString(2 * cm, A4[1] - 2 * cm, "Версії бібліотек:")
    c.setFont("DejaVuSans", 12)
    y = A4[1] - 3 * cm
    for lib, ver in libs.items():
        logger.debug(f"{lib}: {ver}")
        y = safe_draw_string(c, 2 * cm, y, f"{lib}: {ver}")

    # Час виконання
    logger.info("➡️ Додаємо час виконання...")
    exec_time = payload.get("execution_time", "")
    logger.debug(f"execution_time: {exec_time}")
    c.showPage()
    c.setFont("DejaVuSans-Bold", 14)
    c.drawString(2 * cm, A4[1] - 2 * cm, "Час виконання:")
    c.setFont("DejaVuSans", 12)
    c.drawString(2 * cm, A4[1] - 3 * cm, exec_time)

    # Завершення
    logger.info("➡️ Завершуємо формування PDF...")
    c.save()
    buffer.seek(0)

    logger.info("✅ Звіт сформовано успішно!")

    return StreamingResponse(buffer, media_type="application/pdf")

@router.get("/logs")
async def get_logs(type: str = Query(...)):
    file_map = {
        "app_info": "logs/app_info.log",
        "app_debug": "logs/app_debug.log",
        "js_info": "logs/js_info.log",
        "js_debug": "logs/js_debug.log",
    }
    file_path = file_map.get(type)
    if not file_path or not os.path.exists(file_path):
        return {"lines": ["❌ Лог-файл не знайдено"]}
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Вивід перших 10 рядків у консоль для перевірки
    print("=== Перші 10 рядків лог-файлу:", file_path, "===")
    for line in lines[:10]:
        print(line.strip())

    return {"lines": [line.strip() for line in lines]}

@router.get("/logs_page", response_class=HTMLResponse)
async def show_logs_page(request: Request):
    """
    Ендпоінт для відображення сторінки порівняння моделей.
    """
    logger.info("Відкрито сторінку перегляду логів")
    return templates.TemplateResponse(
        "professional/logs.html",
        {"request": request}
    )
