import os
import pandas as pd
from logging_config import logger

from TEST_PROF_ALGRS.regression_algs.main import main as regression_main
from TEST_PROF_ALGRS.classification_algs.main import main as classification_main
from TEST_PROF_ALGRS.clustering_algs.main import main as clustering_main


def run_prediction_task(file_path: str = None, target: str = None, df: pd.DataFrame = None,
                        model_name: str = None, params: dict = None, task_type: str = None):
    """
    Функція для запуску задачі прогнозування.

    Приймає:
    - file_path: шлях до CSV файлу (якщо df не передано)
    - target: назва цільової змінної
    - df: DataFrame із даними
    - model_name: назва моделі
    - params: словник параметрів моделі
    - task_type: тип задачі ("regression", "classification", "clustering")

    Виконує:
    1. Завантаження датасету (якщо df не передано)
    2. Формування словника задачі
    3. Виклик відповідної main-функції залежно від task_type
    4. Повертає словник із результатами (pipeline/model, метрики, ознаки, графіки тощо)
    """

    # Логування входу у функцію
    logger.info("==================================")
    logger.info("=== Запуск функції run_prediction_task ===")
    logger.info("==================================")

    # Логування прийнятих параметрів
    logger.debug(f"file_path: {file_path}")
    logger.debug(f"target: {target}")
    logger.debug(f"df: {'передано' if df is not None else 'None'}")
    logger.debug(f"model_name: {model_name}")
    logger.debug(f"params: {params}")
    logger.debug(f"task_type: {task_type}")

    # Якщо df не передано → читаємо з файлу
    if df is None:
        if not file_path or not os.path.exists(file_path):
            logger.error(f"Файл '{file_path}' не знайдено.")
            raise FileNotFoundError(f"❌ Файл '{file_path}' не знайдено.")
        try:
            df = pd.read_csv(file_path)
            logger.info(f"✅ Датасет успішно завантажено з файлу {file_path}. Розмір: {df.shape}")
        except Exception as e:
            logger.error(f"Помилка при читанні файлу: {e}")
            raise ValueError(f"❌ Помилка при читанні файлу: {e}")
    else:
        logger.info(f"✅ Датасет успішно передано. Розмір: {df.shape}")

    # === Формування словника задачі ===
    task_dict = {
        "df": df,
        "target": target,
        "model_name": model_name,
        "params": params or {}
    }
    logger.debug(f"Словник задачі сформовано: {list(task_dict.keys())}")

    # === Виклик відповідної main-функції ===
    if task_type == "regression":
        logger.info("Запуск regression_main...")
        pipeline, metrics, features, model_params, split_params, plots, input_template = regression_main(task_dict)

        # Логування результатів
        logger.debug("=== Результати regression_main ===")
        logger.debug(f"pipeline: {pipeline}")
        logger.debug(f"metrics: {metrics}")
        logger.debug(f"features: {features}")
        logger.debug(f"model_params: {model_params}")
        logger.debug(f"split_params: {split_params}")
        if plots:
            for name, img in plots.items():
                logger.debug(f"plots -> {name}: {img[:10]}...")
        logger.debug(f"input_template: {input_template}")

        return {
            "task_type": "regression",
            "pipeline": pipeline,
            "metrics": metrics,
            "features": features,
            "model_params": model_params,
            "split_params": split_params,
            "plots": plots,
            "input_template": input_template
        }

    elif task_type == "classification":
        logger.info("Запуск classification_main...")
        pipeline, metrics, features, model_params, split_params, plots, input_template, encoder = classification_main(
            task_dict)

        # Логування результатів
        logger.debug("=== Результати classification_main ===")
        logger.debug(f"pipeline: {pipeline}")
        logger.debug(f"metrics: {metrics}")
        logger.debug(f"features: {features}")
        logger.debug(f"model_params: {model_params}")
        logger.debug(f"split_params: {split_params}")
        if plots:
            for name, img in plots.items():
                logger.debug(f"plots -> {name}: {img[:10]}...")
        logger.debug(f"input_template: {input_template}")
        logger.debug(f"encoder: {encoder}")

        return {
            "task_type": "classification",
            "pipeline": pipeline,
            "metrics": metrics,
            "features": features,
            "model_params": model_params,
            "split_params": split_params,
            "plots": plots,
            "input_template": input_template,
            "encoder": encoder
        }

    elif task_type == "clustering":
        logger.info("Запуск clustering_main...")
        model, labels, plots, input_template, X_train = clustering_main(task_dict)

        # Логування результатів
        logger.debug("=== Результати clustering_main ===")
        logger.debug(f"model: {model}")
        logger.debug(f"labels: {labels}")
        if plots:
            for name, img in plots.items():
                logger.debug(f"plots -> {name}: {img[:10]}...")
        logger.debug(f"input_template: {input_template}")
        logger.debug(f"X_train: {X_train.shape if X_train is not None else 'None'}")

        return {
            "task_type": "clustering",
            "model": model,
            "labels": labels,
            "plots": plots,
            "input_template": input_template,
            "X_train": X_train,
            "metrics": {"clusters": len(set(labels))}
        }

    else:
        logger.error(f"❌ Невідомий тип задачі: {task_type}")
        raise ValueError(f"❌ Невідомий тип задачі: {task_type}")

    logger.info("==========================================")
    logger.info("=== Кінець функції run_prediction_task ===")
    logger.info("==========================================")