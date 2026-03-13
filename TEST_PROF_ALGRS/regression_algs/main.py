from logging_config import logger
from TEST_PROF_ALGRS.regression_algs.models import (
    linear_model,
    random_forest_model,
    gradient_boosting_model,
    xgboost_model
)

def main(task_dict: dict):
    """
    Основна функція для задач регресії.
    Приймає словник task_dict:
    {
        "df": DataFrame,
        "target": str,
        "model_name": str,
        "params": dict
    }
    """

    # Логування входу у функцію
    logger.info("   ===============================")
    logger.info("=== Запуск функції regression_main ===")
    logger.info("   ===============================")

    df = task_dict["df"]
    target = task_dict["target"]
    selected_model = task_dict["model_name"]
    params = task_dict.get("params", {})

    # Логування прийнятих параметрів
    logger.debug(f"target: {target}")
    logger.debug(f"selected_model: {selected_model}")
    logger.debug(f"params: {params}")
    logger.debug(f"df shape: {df.shape}")

    # === Виклик потрібної моделі ===
    if selected_model == "linear_regression":
        pipeline, metrics, features, model_params, split_params, plots, input_template = linear_model.train(df, target, params)
    elif selected_model == "random_forest":
        pipeline, metrics, features, model_params, split_params, plots, input_template = random_forest_model.train(df, target, params)
    elif selected_model == "gradient_boosting":
        pipeline, metrics, features, model_params, split_params, plots, input_template = gradient_boosting_model.train(df, target, params)
    elif selected_model == "xgboost":
        pipeline, metrics, features, model_params, split_params, plots, input_template = xgboost_model.train(df, target, params)
    else:
        logger.error(f"❌ Невідома модель: {selected_model}")
        raise ValueError(f"❌ Невідома модель: {selected_model}")

    # === Логування результатів ===
    logger.info("✅ Модель натренована успішно!")

    # Загальна інформація про повернені змінні
    logger.info(
        "Повернено такі змінні: метрики, ознаки моделі, параметри моделі, параметри train/test split, графіки, шаблон вводу")

    # Метрики
    logger.debug("📊 Метрики:")
    for k, v in metrics.items():
        logger.debug(f"{k}: {v:.4f}")

    # Ознаки
    logger.debug("📌 Використані ознаки:")
    logger.debug(f"{features}")

    # Параметри моделі
    logger.debug("⚙️ Параметри моделі:")
    for k, v in model_params.items():
        logger.debug(f"{k}: {v}")

    # Параметри train/test split
    logger.debug("⚙️ Параметри train/test split:")
    for k, v in split_params.items():
        logger.debug(f"{k}: {v}")

    # Графіки
    logger.debug("🖼️ Доступні графіки:")
    for name, img in plots.items():
        logger.debug(f"{name}: {img[:10]}...")

    # Шаблон вводу
    logger.debug("📋 Шаблон вводу для прогнозу:")
    for feat, desc in input_template.items():
        logger.debug(f"{feat}: {desc}")

    # Логування виходу з функції
    logger.info("   ===============================")
    logger.info("=== Кінець функції regression_main ===")
    logger.info("   ===============================")

    return pipeline, metrics, features, model_params, split_params, plots, input_template
