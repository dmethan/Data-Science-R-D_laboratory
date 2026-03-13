from sklearn.neighbors import KNeighborsClassifier

from TEST_PROF_ALGRS.clustering_algs.models import (
    kmeans_model,
    dbscan_model,
    gmm_model,
    optics_model,
    mean_shift_model,
    spectral_model,
    agglomerative_model
)

from logging_config import logger

def main(task_dict: dict):
    """
    Основна функція для задач кластеризації.
    Приймає словник task_dict:
    {
        "df": DataFrame,
        "target": str або None,
        "model_name": str,
        "params": dict
    }
    """

    logger.info("=== Запуск функції clustering_main ===")

    df = task_dict["df"]
    target = task_dict.get("target", None)
    selected_model = task_dict["model_name"]
    params = task_dict.get("params", {})

    logger.debug(f"target: {target}")
    logger.debug(f"selected_model: {selected_model}")
    logger.debug(f"params: {params}")
    logger.debug(f"df shape: {df.shape}")

    # === Виклик потрібної моделі ===
    if selected_model == "kmeans":
        model, labels, plots, input_template, X_train = kmeans_model.train(df, target, params=params)
    elif selected_model == "dbscan":
        model, labels, plots, input_template, X_train = dbscan_model.train(df, target, params=params)
    elif selected_model == "gmm":
        model, labels, plots, input_template, X_train = gmm_model.train(df, target, params=params)
    elif selected_model == "optics":
        model, labels, plots, input_template, X_train = optics_model.train(df, target, params=params)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, labels)
        model = knn
    elif selected_model == "mean_shift":
        model, labels, plots, input_template, X_train = mean_shift_model.train(df, target, params=params)
    elif selected_model == "spectral":
        model, labels, plots, input_template, X_train = spectral_model.train(df, target, params=params)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, labels)
        model = knn
    elif selected_model == "agglomerative":
        model, labels, plots, input_template, X_train = agglomerative_model.train(df, target, params=params)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, labels)
        model = knn
    else:
        logger.error(f"❌ Невідома модель: {selected_model}")
        raise ValueError(f"❌ Невідома модель: {selected_model}")

    # === Логування результатів ===
    logger.info("✅ Модель натренована успішно!")
    logger.info(f"📊 Кількість кластерів: {len(set(labels))}")
    logger.info("Повернено змінні: model, labels, plots, input_template, X_train")

    logger.debug(f"labels: {labels}")
    logger.debug(f"X_train shape: {X_train.shape}")
    logger.debug(f"features: {list(X_train.columns)}")
    if plots:
        for name, img in plots.items():
            logger.debug(f"plots -> {name}: {img[:10]}...")
    logger.debug(f"input_template: {input_template}")

    logger.info("=== Кінець функції clustering_main ===")

    return model, labels, plots, input_template, X_train
