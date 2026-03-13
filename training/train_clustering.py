import io
import base64
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from logging_config import logger

def train_cluster_model(df: pd.DataFrame, target: str = None):
    """
    Тренування моделі KMeans кластеризації.
    Повертає пакет моделі, метрики, графіки, список ознак, шаблон вводу та функцію prepare_user_input.
    """

    logger.info("=== Запуск функції train_cluster_model ===")

    # 1. Попередня обробка
    df = df.sample(min(10000, len(df)))
    df = df.fillna("missing")
    logger.info("Дані очищено та підготовлено")
    logger.debug(f"df shape: {df.shape}, target: {target}")

    if target and target in df.columns:
        X = df.drop(columns=[target])
    else:
        X = df.copy()
    X = pd.get_dummies(X, drop_first=True)

    # 2. Кореляційна матриця
    numeric_features = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_features.corr()
    logger.info("Кореляційна матриця побудована")

    # 3. Вибір ознак
    threshold = 0.3
    min_features = 5
    if target and target in corr_matrix.columns:
        corr_target = corr_matrix[target].drop(target)
        positive_corr = corr_target[corr_target > 0]
        important_features = positive_corr[positive_corr >= threshold].index.tolist()
        if len(important_features) == 0:
            important_features = positive_corr.sort_values(ascending=False).head(min_features).index.tolist()
        elif len(important_features) < min_features:
            important_features = positive_corr.index.tolist()
        X = df[important_features]
    else:
        important_features = numeric_features.columns.tolist()
        X = numeric_features
    logger.info("Вибрано важливі ознаки")
    logger.debug(f"important_features: {important_features}")

    # 4. Нормалізація
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Дані нормалізовано")

    # 5. Метод "лікоть"
    wcss = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        kmeans_temp.fit(X_scaled)
        wcss.append(kmeans_temp.inertia_)
    logger.info("Метод 'лікоть' виконано")

    # Вибір оптимального k
    diffs = np.diff(wcss)
    optimal_k = np.argmin(diffs) + 2
    logger.info(f"Оптимальна кількість кластерів: {optimal_k}")

    # 6. Запуск KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters
    logger.info("Модель KMeans натренована")

    # 7. Збереження моделі
    model_package = {
        "kmeans": kmeans,
        "scaler": scaler,
        "features": important_features
    }

    os.makedirs("saved_basic_models", exist_ok=True)
    filepath = os.path.join("saved_basic_models", "kmeans_model.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model_package, f)
    logger.info("Модель збережена у файл kmeans_model.pkl")

    # 8. Графіки у форматі base64
    plots = {}

    def save_plot_to_base64():
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Кореляційна матриця
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Матриця кореляції (всі числові ознаки)")
    plots["correlation_matrix_plot"] = save_plot_to_base64()

    # Кореляція ознак з цільовою
    if target and target in corr_matrix.columns:
        plt.figure(figsize=(8, 12))
        sns.heatmap(corr_matrix[[target]].sort_values(by=target, ascending=False),
                    cmap="coolwarm", center=0, annot=True)
        plt.title("Кореляція ознак з цільовою змінною")
        plots["target_correlation_plot"] = save_plot_to_base64()

    # Elbow метод
    plt.figure(figsize=(8, 6))
    plt.plot(K_range, wcss, marker="o", linestyle="--", color="blue")
    plt.title("Метод 'лікоть'")
    plt.xlabel("Кількість кластерів (k)")
    plt.ylabel("WCSS")
    plots["elbow_method_plot"] = save_plot_to_base64()

    # PCA візуалізація
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.6)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Візуалізація кластерів (PCA)")
    plt.colorbar(label="Cluster")
    plots["pca_clusters_plot"] = save_plot_to_base64()

    # Heatmap середніх значень ознак
    cluster_means = df.groupby("Cluster")[important_features].mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Середні значення ознак по кластерах")
    plots["cluster_means_plot"] = save_plot_to_base64()

    # Barplot розподілу кластерів
    cluster_counts = df["Cluster"].value_counts().sort_index()
    cluster_df = cluster_counts.reset_index()
    cluster_df.columns = ["Cluster", "Count"]
    plt.figure(figsize=(8, 6))
    sns.barplot(data=cluster_df, x="Cluster", y="Count", hue="Cluster", palette="viridis", legend=False)
    plt.title("Розподіл кількості об'єктів у кластерах")
    plots["cluster_distribution_plot"] = save_plot_to_base64()

    logger.info("Графіки побудовано")

    # 9. Шаблон вводу
    def generate_input_template(features: list):
        template = {}
        examples = {}
        categorical_features = [col for col in features if "_" in col]
        numeric_features = [col for col in features if "_" not in col]
        for col in categorical_features:
            base, value = col.split("_", 1)
            if base not in examples:
                examples[base] = []
            examples[base].append(value)
        for feat in numeric_features:
            template[feat] = "Введіть число"
        for base, vals in examples.items():
            template[base] = f"Можливі значення: {vals}"
        return template

    input_template = generate_input_template(important_features)
    logger.info("Шаблон вводу сформовано")

    # 10. Функція для перетворення вводу користувача
    def prepare_user_input(user_input: dict, features: list) -> pd.DataFrame:
        values = {feat: 0 for feat in features}
        for feat, val in user_input.items():
            if feat in features:
                values[feat] = val
            else:
                for f in features:
                    if f.startswith(feat + "_"):
                        values[f] = 1 if f == f"{feat}_{val}" else 0
        return pd.DataFrame([values], columns=features)

    metrics = {"n_clusters": int(optimal_k)}
    logger.info("=== Кінець функції train_cluster_model ===")

    return model_package, metrics, plots, important_features, input_template, prepare_user_input
