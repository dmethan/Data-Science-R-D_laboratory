import io, base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import os
import pickle
from logging_config import logger

def load_model(model_name: str, folder: str = "saved_models"):
    """
    Завантажує модель (регресія або класифікація) із .pkl файлу.
    Повертає pipeline та список ознак.
    """

    logger.info("=== Запуск функції load_model ===")

    filepath = os.path.join(folder, f"{model_name}.pkl")
    logger.debug(f"Параметри: model_name={model_name}, folder={folder}, filepath={filepath}")

    if not os.path.exists(filepath):
        logger.error(f"❌ Файл {filepath} не знайдено")
        raise FileNotFoundError(f"❌ Файл {filepath} не знайдено")

    with open(filepath, "rb") as f:
        model_package = pickle.load(f)

    pipeline = model_package["pipeline"]
    features = model_package["features"]

    logger.info(f"✅ Модель '{model_name}' завантажена з {filepath}")
    logger.info("Повернено змінні: pipeline, features")
    logger.debug(f"features: {features}")

    logger.info("=== Кінець функції load_model ===")
    return pipeline, features


def save_model(model_package, model_name: str, folder: str = "saved_models"):
    """
    Зберігає модель у форматі .pkl з фіксованою назвою.
    Кожен новий запуск перезаписує попередній файл.
    """

    logger.info("=== Запуск функції save_model ===")

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{model_name}.pkl")
    logger.debug(f"Параметри: model_name={model_name}, folder={folder}, filepath={filepath}")

    with open(filepath, "wb") as f:
        pickle.dump(model_package, f)

    logger.info(f"✅ Модель '{model_name}' збережена у {filepath}")
    logger.info("Повернено змінну: filepath")

    logger.info("=== Кінець функції save_model ===")
    return filepath

# === Опис параметрів для моделей ===
PARAM_DESCRIPTIONS = {
    "logistic_regression": {
        "l1_ratio": {
            "0": "Чиста L2-регуляризація (Ridge)",
            "1": "Чиста L1-регуляризація (Lasso)",
            "0.5": "Комбінація L1 та L2 (ElasticNet)"
        },
        "C": {
            "мале значення": "Сильна регуляризація",
            "велике значення": "Слабка регуляризація",
            "inf": "Без регуляризації"
        },
        "solver": {
            "lbfgs": "Оптимізатор LBFGS (рекомендовано для багатокласових задач)",
            "liblinear": "Працює лише для бінарної класифікації",
            "saga": "Підтримує L1, L2 та ElasticNet"
        }
    },
    "decision_tree": {
        "criterion": {
            "gini": "Індекс Джині",
            "entropy": "Ентропія"
        }
    },
    "random_forest": {
        "criterion": {
            "gini": "Індекс Джині",
            "entropy": "Ентропія"
        }
    },
    "svm": {
        "kernel": {
            "linear": "Лінійне ядро",
            "poly": "Поліноміальне ядро",
            "rbf": "Радіальне базисне ядро (рекомендовано)",
            "sigmoid": "Сигмоїдне ядро"
        }
    }
}


# === Запит параметра ===
def ask_param(prompt, default, cast_func, min_val=None, max_val=None, options=None, descriptions=None):
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
            logger.warning(f"Менше мінімального ({min_val}), використано {default}")
            return default
        if max_val is not None and val > max_val:
            logger.warning(f"Більше максимального ({max_val}), використано {default}")
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


# === Підготовка датафрейму ===
def prepare_dataframe(df: pd.DataFrame, target: str):
    logger.info("=== Запуск функції prepare_dataframe ===")

    df = df.fillna(0)
    encoder = LabelEncoder()
    df[target] = encoder.fit_transform(df[target])
    y = df[target]

    X = df.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=True)

    y_encoded = pd.get_dummies(y, drop_first=True)
    numeric_features = pd.concat([X, y_encoded], axis=1)
    corr_matrix = numeric_features.corr()

    threshold = 0.3
    min_features = 6
    corr_target = corr_matrix[y_encoded.columns[0]].drop(y_encoded.columns[0])
    positive_corr = corr_target[corr_target > 0]

    important_features = positive_corr[positive_corr >= threshold].index.tolist()
    if len(important_features) == 0:
        important_features = positive_corr.sort_values(ascending=False).head(min_features).index.tolist()
    elif len(important_features) < min_features:
        important_features = positive_corr.index.tolist()

    X = X[important_features]

    logger.info("Повернено змінні: X, y, important_features, corr_matrix, encoder")
    logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.debug(f"important_features: {important_features}")
    logger.info("=== Кінець функції prepare_dataframe ===")

    return X, y, important_features, corr_matrix, encoder


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


# === Генерація шаблону вводу ===
def generate_input_template(features: list):
    logger.info("=== Запуск функції generate_input_template ===")

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


# === Оцінка моделі ===
def evaluate_model(pipeline, X_train, X_test, y_train, y_test, important_features, corr_matrix, target):
    logger.info("=== Запуск функції evaluate_model ===")

    y_pred = pipeline.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted")
    }

    plots = {}
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Матриця помилок (Confusion Matrix)")
    plt.xlabel("Передбачені класи")
    plt.ylabel("Фактичні класи")
    plots["confusion_matrix_plot"] = save_plot_to_base64()

    input_template = generate_input_template(important_features)

    logger.info("Повернено змінні: metrics, plots, input_template")
    logger.debug(f"metrics: {metrics}")
    logger.debug(f"plots keys: {list(plots.keys())}")
    logger.debug(f"input_template: {input_template}")
    logger.info("=== Кінець функції evaluate_model ===")

    return metrics, plots, input_template
