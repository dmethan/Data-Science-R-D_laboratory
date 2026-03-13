import os
import pandas as pd
from training.train_regression import train_regression_model
from training.train_classification import train_classification_model
from training.train_clustering import train_cluster_model
from logging_config import logger


# === Основна функція ===
def run_prediction_task(file_path: str = None, target: str = None, df: pd.DataFrame = None):
    """
    Запуск задачі прогнозування:
    - Якщо target не задано → кластеризація
    - Якщо target числовий з малою кількістю унікальних значень → класифікація
    - Якщо target числовий з великою кількістю унікальних значень → регресія
    - Якщо target категоріальний → класифікація
    """

    logger.info("=== Запуск функції run_prediction_task ===")

    # Якщо df не передано → читаємо з файлу
    if df is None:
        if not file_path or not os.path.exists(file_path):
            logger.error(f"❌ Файл '{file_path}' не знайдено.")
            raise FileNotFoundError(f"❌ Файл '{file_path}' не знайдено.")
        try:
            df = pd.read_csv(file_path)
            logger.info(f"✅ Датасет успішно завантажено з {file_path}")
        except Exception as e:
            logger.error(f"❌ Помилка при читанні файлу: {e}")
            raise ValueError(f"❌ Помилка при читанні файлу: {e}")

    logger.info(f"Розмір датасету: {df.shape}")

    # Якщо цільова змінна не задана → кластеризація
    if target is None:
        logger.info("⚡ Цільова змінна не задана → задача кластеризації.")
        return train_cluster_model(df)

    if target not in df.columns:
        logger.error(f"❌ Колонка '{target}' відсутня у датасеті.")
        raise ValueError(f"❌ Колонка '{target}' відсутня у датасеті. Доступні колонки: {list(df.columns)}")

    target_dtype = df[target].dtype
    unique_values = df[target].nunique()
    logger.info(f"ℹ Тип даних цільової змінної '{target}': {target_dtype}, унікальних значень: {unique_values}")

    # Визначення типу задачі
    if pd.api.types.is_numeric_dtype(df[target]):
        if unique_values <= 10:
            logger.info("⚡ Визначено задачу класифікації (числова змінна з малим числом унікальних значень).")
            return train_classification_model(df, target)
        else:
            logger.info("⚡ Визначено задачу регресії.")
            return train_regression_model(df, target)
    else:
        logger.info("⚡ Визначено задачу класифікації.")
        return train_classification_model(df, target)

    logger.info("=== Кінець функції run_prediction_task ===")

