import io
import pandas as pd

from fastapi import APIRouter, Request, UploadFile, File, Body, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from training.task_detector import run_prediction_task
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from logging_config import logger

router = APIRouter(prefix="/basic_mode", tags=["Basic Mode"])
templates = Jinja2Templates(directory="templates")

basic_state = {
    "dataframe": None,
    "model": None,
    "task_type": None,
    "target_column": None,
    "feature_columns": None
}

@router.get("/", response_class=HTMLResponse)
async def basic_mode_home(request: Request):
    logger.debug("=== Запуск basic_mode_home ===")
    response = templates.TemplateResponse(
        "basic/basic_mode_index.html",
        {"request": request}
    )
    logger.debug("=== Кінець basic_mode_home ===")
    return response

# ------------------------------
# 2️⃣ Завантаження CSV
# ------------------------------
@router.post("/upload", response_class=HTMLResponse)
async def upload_csv(
    request: Request,
    file: UploadFile = File(...)
):
    logger.info("=== Запуск upload_csv ===")

    if not file.filename.endswith(".csv"):
        logger.error("❌ Завантажено не CSV файл")
        raise HTTPException(status_code=400, detail="Потрібен CSV файл")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    basic_state["dataframe"] = df

    preview = df.head(10)
    columns = df.columns.tolist()
    dtypes = {col: str(df[col].dtype) for col in columns}

    logger.info(f"✅ CSV успішно завантажено. Розмір: {df.shape}")
    logger.debug(f"Колонки: {columns}")

    return templates.TemplateResponse(
        "basic/basic_mode_index.html",
        {
            "request": request,
            "table": preview.to_dict(orient="records"),
            "columns": columns,
            "dtypes": dtypes
        }
    )

# ------------------------------
# 3️⃣ Навчання моделі
# ------------------------------
@router.post("/train")
async def train_model(payload: dict = Body(...)):
    logger.info("=== Запуск train_model ===")

    df = basic_state["dataframe"]
    target_column = payload.get("target")

    if df is None:
        logger.error("❌ Немає даних для навчання")
        raise HTTPException(status_code=400, detail="Немає даних для навчання")

    if target_column and target_column not in df.columns:
        logger.error(f"❌ Колонка '{target_column}' відсутня у датасеті")
        raise HTTPException(status_code=400, detail=f"Колонка '{target_column}' відсутня у датасеті")

    try:
        model, metrics, plots, features, input_template, prepare_input = run_prediction_task(
            df=df,
            target=target_column if target_column else None
        )
    except Exception as e:
        logger.error(f"❌ Помилка при навчанні моделі: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Зберігаємо стан
    basic_state["model"] = model
    basic_state["target_column"] = target_column if target_column else None
    basic_state["feature_columns"] = features
    basic_state["prepare_input"] = prepare_input

    # Визначаємо тип задачі
    if isinstance(model, Pipeline):  # якщо це Pipeline
        last_estimator = model.steps[-1][1]
        if isinstance(last_estimator, LinearRegression):
            task_type = "regression"
            metric_name = "R2"
            metric_value = metrics.get("R2")
        elif isinstance(last_estimator, LogisticRegression):
            task_type = "classification"
            metric_name = "accuracy"
            metric_value = metrics.get("accuracy")
        elif isinstance(last_estimator, KMeans):
            task_type = "clustering"
            metric_name = "clusters"
            metric_value = metrics.get("n_clusters", "Модель кластеризації побудована")
        else:
            task_type = "unknown"
            metric_name = "info"
            metric_value = "Невідомий тип моделі"
    elif isinstance(model, dict) and "kmeans" in model:  # якщо це кластеризація через model_package
        task_type = "clustering"
        metric_name = "clusters"
        metric_value = int(metrics.get("n_clusters", "Модель кластеризації побудована"))
    elif isinstance(model, KMeans):  # якщо повертається чистий KMeans
        task_type = "clustering"
        metric_name = "clusters"
        metric_value = metrics.get("n_clusters", "Модель кластеризації побудована")
    else:
        task_type = "unknown"
        metric_name = "info"
        metric_value = "Невідомий тип моделі"

    logger.info(f"✅ Модель успішно навчена. Тип задачі: {task_type}, метрика: {metric_name}={metric_value}")

    return {
        "message": "✅ Модель успішно навчена",
        "target_column": target_column if target_column else None,
        "task": task_type,
        "features": features,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "plots": plots
    }

# ------------------------------
# 4️⃣ Прогноз нового значення
# ------------------------------
@router.post("/predict-value")
async def predict_value(payload: dict = Body(...)):
    logger.info("=== Запуск predict_value ===")

    model = basic_state.get("model")
    features = basic_state.get("feature_columns")
    prepare_input = basic_state.get("prepare_input")

    if model is None or features is None or prepare_input is None:
        logger.error("❌ Модель ще не навчена")
        raise HTTPException(status_code=400, detail="Модель ще не навчена")

    user_values = payload.get("values")
    if not user_values:
        logger.error("❌ Немає даних для прогнозу")
        raise HTTPException(status_code=400, detail="Немає даних для прогнозу")

    try:
        new_data = prepare_input(user_values, features)

        if isinstance(model, dict) and "kmeans" in model:
            scaler = model["scaler"]
            kmeans = model["kmeans"]
            new_data_scaled = scaler.transform(new_data)
            cluster = int(kmeans.predict(new_data_scaled)[0])
            logger.info(f"➡️ Кластеризація — передбачений кластер: {cluster}")
            result = {"prediction": f"Об'єкт належить до кластера {cluster}"}

        elif hasattr(model, "predict"):
            prediction = model.predict(new_data)
            prediction_value = int(prediction[0]) if hasattr(prediction[0], "item") else prediction[0]
            logger.info(f"➡️ Класифікація/Регресія — передбачене значення: {prediction_value}")
            result = {"prediction": prediction_value}

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(new_data)[0]
                probabilities_list = [float(p) for p in probabilities]
                logger.info(f"➡️ Класифікація — ймовірності класів: {probabilities_list}")
                result["probabilities"] = probabilities_list
        else:
            logger.error("❌ Модель не підтримує прогнозування")
            raise Exception("Модель не підтримує прогнозування")

        logger.info("✅ Прогноз виконано успішно")
        return result

    except Exception as e:
        logger.error(f"❌ Помилка при прогнозуванні: {e}")
        raise HTTPException(status_code=500, detail=str(e))
