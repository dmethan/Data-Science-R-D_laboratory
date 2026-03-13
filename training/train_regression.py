import io
import base64
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logging_config import logger
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def train_regression_model(df: pd.DataFrame, target: str):
    """
    Тренування моделі LinearRegression.
    Повертає pipeline, метрики, графіки, список ознак, шаблон вводу та функцію prepare_user_input.
    """

    logger.info("=== Запуск функції train_regression_model ===")

    # 1. Попередня обробка
    df = df.fillna(0)
    y = df[target]
    X = df.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=True)
    logger.info("Дані очищено та закодовано")
    logger.debug(f"df shape: {df.shape}, target: {target}")

    # 2. Кореляційна матриця
    numeric_features = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_features.corr()
    logger.info("Кореляційна матриця побудована")

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
    logger.info("Вибрано важливі ознаки")
    logger.debug(f"important_features: {important_features}")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Дані розділено на train/test")
    logger.debug(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 4. Створення Pipeline (scaler + модель)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    logger.info("Pipeline створено")

    # 5. Навчання моделі
    pipeline.fit(X_train, y_train)
    logger.info("Модель натренована")

    # 6. Метрики
    y_pred = pipeline.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }
    logger.info("Метрики обчислено")
    logger.debug(f"metrics: {metrics}")

    # 7. Збереження моделі у pkl разом зі списком ознак
    model_package = {
        "pipeline": pipeline,
        "features": important_features
    }

    os.makedirs("saved_basic_models", exist_ok=True)
    filepath = os.path.join("saved_basic_models", "linear_model_pipeline.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model_package, f)
    logger.info("Модель збережена у файл linear_model_pipeline.pkl")

    # with open("linear_model_pipeline.pkl", "wb") as f:
    #     pickle.dump(model_package, f)
    # logger.info("Модель збережена у файл linear_model_pipeline.pkl")

    # 8. Графіки у форматі base64
    plots = {}

    def save_plot_to_base64():
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Матриця кореляції
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Матриця кореляції (всі числові ознаки)")
    plots["correlation_matrix_plot"] = save_plot_to_base64()

    # Вертикальна матриця кореляції
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

    logger.info("Графіки побудовано")

    # 9. Автоматичний шаблон вводу
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

    logger.info("=== Кінець функції train_regression_model ===")

    return pipeline, metrics, plots, important_features, input_template, prepare_user_input
