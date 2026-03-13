import io
import base64
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from logging_config import logger

def train_classification_model(df: pd.DataFrame, target: str):
    """
    Тренування моделі LogisticRegression для класифікації.
    Повертає pipeline, метрики, графіки, список ознак, шаблон вводу та функцію prepare_user_input.
    """

    logger.info("=== Запуск функції train_classification_model ===")

    # 1. Попередня обробка
    df = df.fillna("missing")
    y = df[target]
    X = df.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=True)
    logger.info("Дані очищено та закодовано")
    logger.debug(f"df shape: {df.shape}, target: {target}")

    # 2. Кореляційна матриця
    y_encoded = pd.get_dummies(y, drop_first=True)
    numeric_features = pd.concat([X, y_encoded], axis=1)
    corr_matrix = numeric_features.corr()
    logger.info("Кореляційна матриця побудована")

    # Вибір ознак
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
    logger.info("Вибрано важливі ознаки")
    logger.debug(f"important_features: {important_features}")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Дані розділено на train/test")
    logger.debug(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 4. Pipeline (scaler + LogisticRegression)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    logger.info("Pipeline створено")

    # 5. Навчання моделі
    pipeline.fit(X_train, y_train)
    logger.info("Модель LogisticRegression натренована")

    # 6. Метрики
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    logger.info("Метрики обчислено")
    logger.debug(f"metrics: {metrics}")

    # 7. Збереження моделі у pkl
    model_package = {
        "pipeline": pipeline,
        "features": important_features
    }
    os.makedirs("saved_basic_models", exist_ok=True)
    filepath = os.path.join("saved_basic_models", "logistic_model_pipeline.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model_package, f)
    logger.info("Модель збережена у файл logistic_model_pipeline.pkl")

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
    plt.title("Матриця кореляції (всі ознаки)")
    plots["correlation_matrix_plot"] = save_plot_to_base64()

    # Кореляція ознак з цільовою
    plt.figure(figsize=(4, 12))
    sns.heatmap(corr_matrix[[y_encoded.columns[0]]].sort_values(by=y_encoded.columns[0], ascending=False),
                cmap="coolwarm", center=0, annot=True)
    plt.title("Кореляція ознак з цільовою змінною")
    plots["target_correlation_plot"] = save_plot_to_base64()

    # Матриця плутанини
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Передбачені класи")
    plt.ylabel("Фактичні класи")
    plt.title("Матриця плутанини")
    plots["confusion_matrix_plot"] = save_plot_to_base64()

    # ROC-криві
    unique_classes = np.unique(y)
    y_prob = pipeline.predict_proba(X_test)
    if len(unique_classes) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=unique_classes[1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", label=f"ROC-крива (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-крива (бінарна класифікація)")
        plt.legend(loc="lower right")
        plots["roc_curve_plot"] = save_plot_to_base64()
    else:
        plt.figure(figsize=(8, 6))
        for i, cls in enumerate(unique_classes):
            fpr, tpr, _ = roc_curve(y_test == cls, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Клас {cls} (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-криві для багатокласової класифікації (One-vs-Rest)")
        plt.legend()
        plots["roc_curve_multiclass_plot"] = save_plot_to_base64()

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

    logger.info("=== Кінець функції train_classification_model ===")

    return pipeline, metrics, plots, important_features, input_template, prepare_user_input
