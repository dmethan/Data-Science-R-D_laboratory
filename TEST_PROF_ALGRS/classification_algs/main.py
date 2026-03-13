from TEST_PROF_ALGRS.classification_algs.models import (
    logistic_model,
    knn_model,
    svm_model,
    gradient_boosting_model,
    decision_tree_model,
    xgboost_model,
    random_forest_model
)
from logging_config import logger

def main(task_dict: dict):
    """
    Основна функція для задач класифікації.
    Приймає словник task_dict:
    {
        "df": DataFrame,
        "target": str,
        "model_name": str,
        "params": dict
    }
    """

    logger.info("=== Запуск функції classification_main ===")

    df = task_dict["df"]
    target = task_dict["target"]
    selected_model = task_dict["model_name"]
    params = task_dict.get("params", {})

    logger.debug(f"target: {target}")
    logger.debug(f"selected_model: {selected_model}")
    logger.debug(f"params: {params}")
    logger.debug(f"df shape: {df.shape}")

    # === Виклик потрібної моделі ===
    if selected_model == "logistic_regression":
        pipeline, metrics, features, model_params, split_params, plots, input_template, encoder = logistic_model.train(df, target, params=params)
    elif selected_model == "decision_tree":
        pipeline, metrics, features, model_params, split_params, plots, input_template, encoder = decision_tree_model.train(df, target, params=params)
    elif selected_model == "random_forest":
        pipeline, metrics, features, model_params, split_params, plots, input_template, encoder = random_forest_model.train(df, target, params=params)
    elif selected_model == "gradient_boosting":
        pipeline, metrics, features, model_params, split_params, plots, input_template, encoder = gradient_boosting_model.train(df, target, params=params)
    elif selected_model == "xgboost":
        pipeline, metrics, features, model_params, split_params, plots, input_template, encoder = xgboost_model.train(df, target, params=params)
    elif selected_model == "svm":
        pipeline, metrics, features, model_params, split_params, plots, input_template, encoder = svm_model.train(df, target, params=params)
    elif selected_model == "knn":
        pipeline, metrics, features, model_params, split_params, plots, input_template, encoder = knn_model.train(df, target, params=params)
    else:
        logger.error(f"❌ Невідома модель: {selected_model}")
        raise ValueError(f"❌ Невідома модель: {selected_model}")

    # === Логування результатів ===
    logger.info("✅ Модель натренована успішно!")
    logger.info("Повернено змінні: pipeline, metrics, features, model_params, split_params, plots, input_template, encoder")
    logger.debug(f"metrics: {metrics}")
    logger.debug(f"features: {features}")
    logger.debug(f"model_params: {model_params}")
    logger.debug(f"split_params: {split_params}")
    logger.debug(f"plots keys: {list(plots.keys())}")
    logger.debug(f"input_template: {input_template}")
    logger.debug(f"encoder: {encoder}")

    logger.info("=== Кінець функції classification_main ===")

    return pipeline, metrics, features, model_params, split_params, plots, input_template, encoder
