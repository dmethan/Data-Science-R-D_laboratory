import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from TEST_PROF_ALGRS.clustering_algs.utils import (
    save_plot_to_base64,
    generate_input_template,
    preprocess_data,
    save_cluster_model
)
from logging_config import logger

def train(df, target: str = None, params: dict = None):
    """
    Тренування моделі DBSCAN.
    Параметри приймаються через словник params.
    """

    logger.info("=== Запуск функції train (DBSCAN) ===")

    # Значення за замовчуванням
    default_params = {
        "eps": 0.5,
        "min_samples": 5,
        "metric": "euclidean"
    }

    # Об’єднання словника користувача з дефолтними параметрами
    if params is not None:
        model_params = {**default_params, **params}
        logger.info("Використано параметри користувача + дефолтні")
    else:
        model_params = default_params
        logger.info("Використано дефолтні параметри")

    logger.debug(f"model_params: {model_params}")

    # Створюємо модель
    model = DBSCAN(**model_params)
    logger.info("Модель DBSCAN створена")

    # === Попередня обробка даних (з обмеженням до 10k рядків) ===
    X, important_features, corr_matrix, X_scaled = preprocess_data(df, target)
    X_scaled_df = pd.DataFrame(X_scaled, columns=important_features)
    logger.info("Попередня обробка даних завершена")
    logger.debug(f"X_scaled_df shape: {X_scaled_df.shape}")
    logger.debug(f"important_features: {important_features}")

    # === Навчання моделі ===
    labels = model.fit_predict(X_scaled_df)
    logger.info("Модель натренована")
    logger.debug(f"labels: {labels}")

    # Збереження моделі
    save_cluster_model(model, "dbscan_model", important_features)
    logger.info("Модель збережена у файл dbscan_model.pkl")

    # Додаємо кластери тільки до тієї вибірки, яка реально використовувалась
    df_limited = df.iloc[:len(labels)].copy()
    df_limited["Cluster"] = labels

    plots = {}

    # --- PCA візуалізація ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled_df)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.6)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Візуалізація кластерів (DBSCAN, PCA)")
    plt.colorbar(label="Cluster")
    plots["pca_clusters_plot"] = save_plot_to_base64()

    # --- Heatmap середніх значень ознак ---
    if len(set(labels)) > 1:  # якщо є кластери, крім шуму
        cluster_means = df_limited.groupby("Cluster")[important_features].mean()
        plt.figure(figsize=(10, 6))
        sns.heatmap(cluster_means, annot=True, cmap="viridis", fmt=".2f")
        plt.title("Середні значення ознак по кластерах (DBSCAN)")
        plots["cluster_means_plot"] = save_plot_to_base64()

    # --- Barplot розподілу кластерів ---
    cluster_counts = df_limited["Cluster"].value_counts().sort_index()
    cluster_df = cluster_counts.reset_index()
    cluster_df.columns = ["Cluster", "Count"]
    plt.figure(figsize=(8, 6))
    sns.barplot(data=cluster_df, x="Cluster", y="Count", hue="Cluster", palette="viridis", legend=False)
    plt.title("Розподіл кількості об'єктів у кластерах (DBSCAN)")
    plots["cluster_distribution_plot"] = save_plot_to_base64()

    # === Шаблон вводу для користувача ===
    input_template = generate_input_template(important_features, df_limited)
    logger.info("Шаблон вводу для користувача сформовано")
    logger.debug(f"input_template: {input_template}")

    logger.info("=== Кінець функції train (DBSCAN) ===")

    return model, labels, plots, input_template, X_scaled_df
