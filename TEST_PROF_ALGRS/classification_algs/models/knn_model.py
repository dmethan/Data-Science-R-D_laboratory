from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from TEST_PROF_ALGRS.classification_algs.utils import (
    evaluate_model,
    prepare_dataframe,
    save_model
)
from logging_config import logger

def train(df, target, params: dict = None):
    """
    Тренування моделі KNeighborsClassifier.
    Параметри приймаються через словник params.
    """

    logger.info("=== Запуск функції train (KNeighborsClassifier) ===")

    # Значення за замовчуванням
    default_params = {
        "n_neighbors": 5,
        "random_state": 42,   # для узгодженості зі split
        "test_split": 0.2
    }

    # Об’єднання словника користувача з дефолтними параметрами
    if params is not None:
        model_params = {**default_params, **params}
        logger.info("Використано параметри користувача + дефолтні")
    else:
        model_params = default_params
        logger.info("Використано дефолтні параметри")

    logger.debug(f"model_params: {model_params}")

    # Витягуємо train/test split окремо
    test_split = model_params.pop("test_split")
    random_state = model_params.pop("random_state")
    split_params = {"test_split": test_split, "random_state": random_state}
    logger.info("Параметри train/test split сформовано")
    logger.debug(f"split_params: {split_params}")

    # Створюємо модель
    model = KNeighborsClassifier(**model_params)
    logger.info("Модель KNeighborsClassifier створена")

    # Підготовка даних
    X, y, important_features, corr_matrix, encoder = prepare_dataframe(df, target)
    logger.info("Дані підготовлено")
    logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.debug(f"important_features: {important_features}")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state
    )
    logger.info("Дані розділено на train/test")
    logger.debug(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Побудова пайплайну
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", model)])
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline натреновано")

    # Оцінка моделі
    metrics, plots, input_template = evaluate_model(
        pipeline, X_train, X_test, y_train, y_test, important_features, corr_matrix, target
    )
    logger.info("Оцінка моделі завершена")
    logger.debug(f"metrics: {metrics}")
    logger.debug(f"plots keys: {list(plots.keys())}")
    logger.debug(f"input_template: {input_template}")

    # Збереження моделі
    model_package = {
        "pipeline": pipeline,
        "features": important_features
    }
    save_model(model_package, "knn_model")
    logger.info("Модель збережена у файл knn_model.pkl")

    logger.info("=== Кінець функції train (KNeighborsClassifier) ===")

    return pipeline, metrics, important_features, model_params, split_params, plots, input_template, encoder
