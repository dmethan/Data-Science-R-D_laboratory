import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, SpectralClustering
import os
import pickle
from logging_config import logger


def load_cluster_model(model_name: str, folder: str = "saved_models"):
    """
    Завантажує кластеризаційну модель із .pkl файлу.
    Повертає модель та список ознак.
    """

    logger.info("=== Запуск функції load_cluster_model ===")

    filepath = os.path.join(folder, f"{model_name}.pkl")
    logger.debug(f"Параметри: model_name={model_name}, folder={folder}, filepath={filepath}")

    if not os.path.exists(filepath):
        logger.error(f"❌ Файл {filepath} не знайдено")
        raise FileNotFoundError(f"❌ Файл {filepath} не знайдено")

    with open(filepath, "rb") as f:
        model_package = pickle.load(f)

    model = model_package["model"]
    features = model_package["features"]

    logger.info(f"✅ Кластеризаційна модель '{model_name}' завантажена з {filepath}")
    logger.info("Повернено змінні: model, features")
    logger.debug(f"features: {features}")

    logger.info("=== Кінець функції load_cluster_model ===")
    return model, features


def save_cluster_model(model, model_name: str, features: list, folder: str = "saved_models"):
    """
    Зберігає кластеризаційну модель у форматі .pkl разом зі списком ознак.
    Кожен новий запуск перезаписує попередній файл.
    """

    logger.info("=== Запуск функції save_cluster_model ===")

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{model_name}.pkl")
    logger.debug(f"Параметри: model_name={model_name}, folder={folder}, filepath={filepath}")

    model_package = {
        "model": model,
        "features": features
    }

    with open(filepath, "wb") as f:
        pickle.dump(model_package, f)

    logger.info(f"✅ Кластеризаційна модель '{model_name}' збережена у {filepath}")
    logger.info("Повернено змінну: filepath")

    logger.info("=== Кінець функції save_cluster_model ===")
    return filepath


# === Опис параметрів для моделей кластеризації ===
PARAM_DESCRIPTIONS = {
    "kmeans": {
        "init": {
            "k-means++": "Ініціалізація центрів за алгоритмом k-means++ (рекомендовано)",
            "random": "Випадкова ініціалізація центрів"
        }
    },
    "agglomerative": {
        "linkage": {
            "ward": "Метод мінімізації дисперсії (рекомендовано)",
            "complete": "Максимальна відстань між точками",
            "average": "Середня відстань",
            "single": "Мінімальна відстань"
        },
        "metric": {
            "euclidean": "Євклідова відстань (рекомендовано)",
            "manhattan": "Манхеттенська відстань",
            "cosine": "Косинусна відстань"
        }
    },
    "dbscan": {
        "metric": {
            "euclidean": "Євклідова відстань (рекомендовано)",
            "manhattan": "Манхеттенська відстань",
            "cosine": "Косинусна відстань"
        }
    },
    "optics": {
        "metric": {
            "euclidean": "Євклідова відстань (рекомендовано)",
            "manhattan": "Манхеттенська відстань",
            "cosine": "Косинусна відстань"
        },
        "cluster_method": {
            "xi": "Метод xi (рекомендовано)",
            "dbscan": "Метод DBSCAN"
        }
    },
    "gmm": {
        "covariance_type": {
            "full": "Повна матриця коваріації (рекомендовано)",
            "tied": "Спільна матриця для всіх компонент",
            "diag": "Діагональна матриця",
            "spherical": "Сферична форма"
        }
    },
    "mean_shift": {
        "cluster_all": {
            True: "Кластеризувати всі точки (рекомендовано)",
            False: "Деякі точки можуть залишитися без кластеру"
        }
    },
    "spectral": {
        "affinity": {
            "rbf": "RBF ядро (рекомендовано)",
            "nearest_neighbors": "Базується на найближчих сусідах",
            "precomputed": "Попередньо обчислена матриця"
        },
        "assign_labels": {
            "kmeans": "Призначення кластерів через KMeans (рекомендовано)",
            "discretize": "Метод дискретизації"
        }
    }
}


# === Функція запитування параметрів ===
def ask_param(prompt, default, cast_func, min_val=None, max_val=None, options=None, descriptions=None):
    """
    Запит параметра з перевіркою та описом.
    - prompt: назва параметра
    - default: рекомендоване значення
    - cast_func: тип (int, float, str, bool)
    - min_val, max_val: межі для числових параметрів
    - options, descriptions: можливі значення для категоріальних параметрів
    """
    logger.info("=== Запуск функції ask_param ===")

    if options and descriptions:
        logger.debug("Можливі значення:")
        for opt, desc in descriptions.items():
            logger.debug(f"{opt}: {desc}")

    val = input(f"{prompt} [рекомендовано {default}]: ")
    if val.strip() == "":
        logger.info(f"Використано значення за замовчуванням: {default}")
        logger.info("=== Кінець функції ask_param ===")
        return default
    try:
        val = cast_func(val)
        if min_val is not None and val < min_val:
            logger.warning(f"Значення менше мінімального ({min_val}), використано {default}")
            logger.info("=== Кінець функції ask_param ===")
            return default
        if max_val is not None and val > max_val:
            logger.warning(f"Значення більше максимального ({max_val}), використано {default}")
            logger.info("=== Кінець функції ask_param ===")
            return default
        logger.info("=== Кінець функції ask_param ===")
        return val
    except Exception:
        if options and val in options:
            logger.info("=== Кінець функції ask_param ===")
            return val
        logger.warning(f"Некоректне значення, використано {default}")
        logger.info("=== Кінець функції ask_param ===")
        return default


# === Збереження графіка у base64 ===
def save_plot_to_base64():
    logger.info("=== Запуск функції save_plot_to_base64 ===")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    logger.info("Повернено змінну: base64 string")
    logger.debug(f"Перші 50 символів: {encoded[:50]}...")
    logger.info("=== Кінець функції save_plot_to_base64 ===")

    return encoded


# === Попередня обробка даних ===
def preprocess_data(df: pd.DataFrame, target: str = None, max_rows: int = 10000, random_state: int = 42):
    """
    Попередня обробка даних:
    1. Обмеження кількості рядків
    2. Заповнення пропусків
    3. Формування X
    4. Кореляційна матриця
    5. Вибір ознак
    6. Масштабування
    """
    logger.info("=== Запуск функції preprocess_data ===")

    # 1. Обмеження кількості рядків
    if len(df) > max_rows:
        logger.warning(f"Датасет має {len(df)} рядків. Використовується випадкова підвибірка з {max_rows} рядків.")
        df = df.sample(max_rows, random_state=random_state).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df = df.fillna("missing")

    # 2. Формування X
    if target and target in df.columns:
        X = df.drop(columns=[target])
    else:
        X = df.copy()

    X = pd.get_dummies(X, drop_first=True)

    # 3. Кореляційна матриця
    numeric_features = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_features.corr()

    # 4. Вибір ознак
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

    # 5. Масштабування
    X_scaled = StandardScaler().fit_transform(X)

    logger.info("Повернено змінні: X, important_features, corr_matrix, X_scaled")
    logger.debug(f"X shape: {X.shape}, important_features: {important_features}")
    logger.debug(f"corr_matrix shape: {corr_matrix.shape}")
    logger.info("=== Кінець функції preprocess_data ===")

    return X, important_features, corr_matrix, X_scaled


# === Генерація шаблону вводу ===
def generate_input_template(features: list, df: pd.DataFrame = None):
    logger.info("=== Запуск функції generate_input_template ===")

    template = {}
    for feat in features:
        if df is not None and feat in df.columns:
            if pd.api.types.is_numeric_dtype(df[feat]):
                template[feat] = "Введіть число"
            else:
                unique_vals = df[feat].dropna().unique().tolist()
                template[feat] = f"Можливі значення: {unique_vals}"
        else:
            template[feat] = "Введіть число"

    logger.info("Повернено змінну: template")
    logger.debug(f"template: {template}")
    logger.info("=== Кінець функції generate_input_template ===")

    return template


# === Підготовка вводу користувача ===
def prepare_user_input(user_input: dict, features: list) -> pd.DataFrame:
    logger.info("=== Запуск функції prepare_user_input ===")

    values = {feat: 0 for feat in features}
    for feat, val in user_input.items():
        if feat in features:
            values[feat] = val
        else:
            for f in features:
                if f.startswith(feat + "_"):
                    values[f] = 1 if f == f"{feat}_{val}" else 0

    df_input = pd.DataFrame([values], columns=features)

    logger.info("Повернено змінну: DataFrame з новим ввідом")
    logger.debug(f"DataFrame shape: {df_input.shape}")
    logger.info("=== Кінець функції prepare_user_input ===")

    return df_input


# === Універсальна функція прогнозування ===
def predict_cluster(model, X_new, X_train=None):
    """
    Універсальна функція прогнозування належності нової точки до кластера.
    """
    logger.info("=== Запуск функції predict_cluster ===")

    # Якщо модель має метод predict (KMeans, GMM)
    if hasattr(model, "predict"):
        try:
            result = model.predict(X_new)
            logger.info("Повернено змінну: result (predict)")
            logger.debug(f"result: {result}")
            logger.info("=== Кінець функції predict_cluster ===")
            return result
        except Exception:
            logger.error("Ця модель не підтримує пряме прогнозування нових точок.")
            raise NotImplementedError("Ця модель не підтримує пряме прогнозування нових точок.")

    # Якщо DBSCAN
    elif isinstance(model, DBSCAN):
        from sklearn.neighbors import NearestNeighbors
        if X_train is None:
            logger.error("Для DBSCAN потрібно передати X_train.")
            raise ValueError("Для DBSCAN потрібно передати X_train.")
        nn = NearestNeighbors(n_neighbors=1).fit(X_train)
        distances, indices = nn.kneighbors(X_new)
        labels = []
        for dist, idx in zip(distances, indices):
            if dist[0] <= model.eps:
                labels.append(model.labels_[idx[0]])
            else:
                labels.append(-1)
        result = np.array(labels)
        logger.info("Повернено змінну: result (DBSCAN)")
        logger.debug(f"result: {result}")
        logger.info("=== Кінець функції predict_cluster ===")
        return result

    # Якщо OPTICS
    elif isinstance(model, OPTICS):
        from sklearn.neighbors import NearestNeighbors
        if X_train is None:
            logger.error("Для OPTICS потрібно передати X_train.")
            raise ValueError("Для OPTICS потрібно передати X_train.")
        nn = NearestNeighbors(n_neighbors=1).fit(X_train)
        distances, indices = nn.kneighbors(X_new)
        labels = []
        for dist, idx in zip(distances, indices):
            labels.append(model.labels_[idx[0]])
        result = np.array(labels)
        logger.info("Повернено змінну: result (OPTICS)")
        logger.debug(f"result: {result}")
        logger.info("=== Кінець функції predict_cluster ===")
        return result

    elif isinstance(model, AgglomerativeClustering):
        logger.error("Agglomerative Clustering не підтримує прогнозування нових точок.")
        raise NotImplementedError("Agglomerative Clustering не підтримує прогнозування нових точок.")

    elif isinstance(model, SpectralClustering):
        logger.error("Spectral Clustering не підтримує прогнозування нових точок.")
        raise NotImplementedError("Spectral Clustering не підтримує прогнозування нових точок.")

    else:
        logger.error("Ця модель не підтримує прогнозування.")
        raise NotImplementedError("Ця модель не підтримує прогнозування.")

