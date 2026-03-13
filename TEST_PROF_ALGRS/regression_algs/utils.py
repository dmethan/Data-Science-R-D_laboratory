import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
import pickle
from logging_config import logger


def load_model(model_name: str, folder: str = "saved_models"):
    """
    Завантажує модель (регресія або класифікація) із .pkl файлу.
    Повертає pipeline та список ознак.
    """

    # Логування входу у функцію
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

    # Логування результатів
    logger.info(f"✅ Модель '{model_name}' завантажена з {filepath}")
    logger.info("Повернено такі змінні: pipeline, features")
    logger.debug(f"features: {features}")

    # Логування виходу з функції
    logger.info("=== Кінець функції load_model ===")

    return pipeline, features


def save_model(model_package, model_name: str, folder: str = "saved_models"):
    """
    Зберігає модель у форматі .pkl з фіксованою назвою.
    Кожен новий запуск перезаписує попередній файл.
    """

    # Логування входу у функцію
    logger.info("=== Запуск функції save_model ===")

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{model_name}.pkl")
    logger.debug(f"Параметри: model_name={model_name}, folder={folder}, filepath={filepath}")

    with open(filepath, "wb") as f:
        pickle.dump(model_package, f)

    # Логування результатів
    logger.info(f"✅ Модель '{model_name}' збережена у {filepath}")
    logger.info("Повернено такі змінні: filepath")

    # Логування виходу з функції
    logger.info("=== Кінець функції save_model ===")

    return filepath


# === Опис параметрів для моделей ===
PARAM_DESCRIPTIONS = {
    "xgboost": {
        "objective": {
            "reg:squarederror": "Класична регресія, мінімізує MSE",
            "reg:logistic": "Логістична регресія для ймовірностей",
            "reg:absoluteerror": "Мінімізує MAE",
            "reg:squaredlogerror": "Квадратична похибка логарифмованих значень",
            "reg:pseudohubererror": "Гладка функція втрат, стійка до викидів",
            "reg:tweedie": "Tweedie distribution (страхування, фінанси)",
            "reg:gamma": "Gamma distribution для позитивних даних"
        }
    },
    "gradient_boosting": {
        "loss": {
            "squared_error": "Класична квадратична похибка",
            "absolute_error": "Абсолютна похибка (MAE)",
            "huber": "Комбінація MSE та MAE, стійка до викидів"
        },
        "criterion": {
            "friedman_mse": "Критерій Фрідмана для розщеплення вузлів",
            "squared_error": "Класична квадратична похибка"
        }
    }
}


# === Запит параметра з перевіркою та описом ===
def ask_param(prompt, default, cast_func, min_val=None, max_val=None, options=None, descriptions=None):
    """
    Запит параметра з перевіркою та описом.
    Повертає значення параметра або default, якщо введено некоректне.
    """
    logger.info("=== Запуск функції ask_param ===")

    if options and descriptions:
        logger.debug("Можливі значення:")
        for opt, desc in descriptions.items():
            logger.debug(f"{opt}: {desc}")

    val = input(f"{prompt} [рекомендовано {default}]: ")
    if val.strip() == "":
        logger.debug(f"Використано значення за замовчуванням: {default}")
        return default
    try:
        val = cast_func(val)
        if min_val is not None and val < min_val:
            logger.warning(f"Значення менше мінімального ({min_val}), використано {default}")
            return default
        if max_val is not None and val > max_val:
            logger.warning(f"Значення більше максимального ({max_val}), використано {default}")
            return default
        return val
    except Exception:
        if options and val in options:
            return val
        logger.warning(f"Некоректне значення, використано {default}")
        return default

    logger.info("=== Кінець функції ask_param ===")


# === Підготовка датафрейму ===
def prepare_dataframe(df: pd.DataFrame, target: str):
    """
    Підготовка датафрейму для моделювання:
    - заповнення пропусків
    - формування X та y
    - вибір важливих ознак за кореляцією
    """
    logger.info("=== Запуск функції prepare_dataframe ===")

    df = df.fillna(0)
    y = df[target]
    X = df.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=True)

    # Кореляційна матриця
    numeric_features = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_features.corr()

    # Вибір ознак
    threshold = 0.3
    min_features = 5
    corr_target = corr_matrix[target].drop(target)
    positive_corr = corr_target[corr_target > 0]
    important_features = positive_corr[positive_corr >= threshold].index.tolist()
    if len(important_features) == 0:
        important_features = positive_corr.sort_values(ascending=False).head(min_features).index.tolist()
    elif len(important_features) < min_features:
        important_features = positive_corr.index.tolist()

    X = df[important_features]

    logger.info("Повернено такі змінні: X, y, important_features, corr_matrix")
    logger.debug(f"important_features: {important_features}")
    logger.debug(f"corr_matrix shape: {corr_matrix.shape}")

    logger.info("=== Кінець функції prepare_dataframe ===")
    return X, y, important_features, corr_matrix


# === Збереження графіка у base64 ===
def save_plot_to_base64():
    """
    Зберігає поточний графік matplotlib у base64.
    """
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
    """
    Генерує шаблон вводу для користувача:
    - для числових ознак: "Введіть число"
    - для категоріальних: список можливих значень
    """
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


# === Підготовка нового вводу користувача ===
def prepare_user_input(user_input: dict, features: list) -> pd.DataFrame:
    """
    Формує DataFrame для нового вводу користувача на основі ознак моделі.
    """
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
    return pd.DataFrame([values], columns=features)


# === Оцінка моделі та побудова графіків ===
def evaluate_model(pipeline, X_train, X_test, y_train, y_test, important_features, corr_matrix, target):
    """
    Оцінює модель за метриками та будує графіки.
    Повертає словник метрик, графіків у base64 та шаблон вводу.
    """
    logger.info("=== Запуск функції evaluate_model ===")

    y_pred = pipeline.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }
    logger.info("Повернено змінну: metrics")
    logger.debug(f"metrics: {metrics}")

    plots = {}

    # Матриця кореляції
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Матриця кореляції (всі числові ознаки)")
    plots["correlation_matrix_plot"] = save_plot_to_base64()

    # Кореляція з цільовою
    plt.figure(figsize=(4, 12))
    sns.heatmap(corr_matrix[[target]].sort_values(by=target, ascending=False),
                cmap="coolwarm", center=0, annot=True)
    plt.title("Кореляція ознак з цільовою змінною")
    plots["target_correlation_plot"] = save_plot_to_base64()

    # Фактичні vs передбачені
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Фактичні значення (y_test)")
    plt.ylabel("Передбачені значення (y_pred)")
    plt.title("Порівняння фактичних та передбачених значень")
    plots["actual_vs_predicted_plot"] = save_plot_to_base64()

    # Розподіл помилок
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True, color="purple")
    plt.xlabel("Помилка (y_test - y_pred)")
    plt.title("Розподіл помилок моделі")
    plots["residuals_distribution_plot"] = save_plot_to_base64()

    # Помилки vs передбачені
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color="green")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Передбачені значення (y_pred)")
    plt.ylabel("Помилка (y_test - y_pred)")
    plt.title("Помилки проти передбачених значень")
    plots["residuals_vs_predicted_plot"] = save_plot_to_base64()

    input_template = generate_input_template(important_features)
    logger.info("Повернено змінні: plots, input_template")
    logger.debug(f"plots keys: {list(plots.keys())}")
    logger.debug(f"input_template: {input_template}")

    logger.info("=== Кінець функції evaluate_model ===")
    return metrics, plots, input_template
