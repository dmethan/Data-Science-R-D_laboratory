/**
 * Файл: basic_mode.js
 * Опис: Логіка для базового режиму ML-сервісу.
 * Містить функції для вибору колонки, тренування моделі та прогнозування.
 * Використовує Winston-логер для запису подій у файл.
 */
import { logger } from "./logger.js";

let selectedColumn = null;
let featureColumns = [];
let taskType = "";

// Імпорт Winston-логера

/**
 * Функція selectColumn
 * Етапи:
 * 1. Знімає виділення з попередньої колонки.
 * 2. Виділяє нову колонку.
 * 3. Оновлює інформацію про вибір у UI.
 * 4. Вмикає кнопку тренування.
 * 5. Логує вибір колонки.
 */
function selectColumn(element, dtype) {
    if (selectedColumn) {
        selectedColumn.classList.remove("selected"); // зняти виділення
    }
    element.classList.add("selected"); // виділити нову колонку
    selectedColumn = element;

    // Оновлення інформації у UI
    document.getElementById("task-info").innerHTML =
        "<strong>Обрана колонка:</strong> " + element.innerText;

    // Показати кнопку тренування
    document.getElementById("train-btn").style.display = "inline-block";

    logger("info", `Колонка '${element.innerText}' обрана для аналізу`);
}

/**
 * Функція trainModel
 * Етапи:
 * 1. Перевіряє, чи вибрано колонку (якщо потрібна).
 * 2. Показує індикатор завантаження.
 * 3. Формує payload для бекенду.
 * 4. Відправляє запит на тренування моделі.
 * 5. Обробляє відповідь: метрики, тип задачі, ознаки.
 * 6. Динамічно створює поля вводу для прогнозу.
 * 7. Логує результат.
 */
async function trainModel(noTarget = false) {
    if (!noTarget && !selectedColumn) {
//        logger.warn("Спроба тренування без вибраної колонки");
        logger("warn", "Спроба тренування без вибраної колонки")
        return;
    }

    document.getElementById("loading").style.display = "block";
    document.getElementById("train-btn").style.display = "none";

    const payload = noTarget ? {} : { target: selectedColumn.innerText };

    try {
        const response = await fetch("/basic_mode/train", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        document.getElementById("loading").style.display = "none";

        if (result.error) {
            logger("error", `Помилка при тренуванні: ${result.error}`)
            alert(result.error);
            return;
        }

        // Збереження стану
        taskType = result.task;
        featureColumns = result.features;

        // Вивід метрик у UI
        document.getElementById("task-info").innerHTML +=
            "<br><strong>Тип задачі:</strong> " + taskType +
            "<br><strong>" + result.metric_name + ":</strong> " + result.metric_value;

        // Генерація полів вводу
        const inputFieldsDiv = document.getElementById("input-fields");
        inputFieldsDiv.innerHTML = "";
        result.features.forEach(f => {
            const container = document.createElement("div");
            container.classList.add("input-container");
            container.innerHTML = `<label>${f}</label><input name="${f}" required>`;
            inputFieldsDiv.appendChild(container);
        });

        document.getElementById("input-form").style.display = "block";
        logger("info", `Модель успішно натренована. Тип задачі: ${taskType}`)

    } catch (e) {
        logger("error", `Помилка при запиті тренування: ${e.message}`)
    }
}

/**
 * Функція trainModelNoTarget
 * Етапи:
 * 1. Показує індикатор завантаження.
 * 2. Відправляє запит на кластеризацію (без target).
 * 3. Обробляє відповідь: метрики, ознаки.
 * 4. Генерує поля вводу.
 * 5. Логує результат.
 */
async function trainModelNoTarget() {
    document.getElementById("loading").style.display = "block";

    try {
        const response = await fetch("/basic_mode/train", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({})
        });

        const result = await response.json();
        document.getElementById("loading").style.display = "none";

        if (result.error) {
            logger("error", `Помилка при кластеризації: ${result.error}`)
            alert(result.error);
            return;
        }

        taskType = result.task;
        featureColumns = result.features;

        document.getElementById("task-info").innerHTML +=
            "<br><strong>Тип задачі:</strong> " + taskType +
            "<br><strong>" + result.metric_name + ":</strong> " + result.metric_value;

        const inputFieldsDiv = document.getElementById("input-fields");
        inputFieldsDiv.innerHTML = "";
        result.features.forEach(f => {
            const container = document.createElement("div");
            container.classList.add("input-container");
            container.innerHTML = `<label>${f}</label><input name="${f}" required>`;
            inputFieldsDiv.appendChild(container);
        });

        document.getElementById("input-form").style.display = "block";
        logger("info", "Модель кластеризації успішно натренована")
    } catch (e) {
        logger("error", `Помилка при запиті кластеризації: ${e.message}`)
    }
}

/**
 * Функція predict
 * Етапи:
 * 1. Збирає дані з форми.
 * 2. Формує payload для бекенду.
 * 3. Відправляє запит на прогноз.
 * 4. Обробляє результат: прогноз, ймовірності, кластерні дані.
 * 5. Виводить результат у UI.
 * 6. Логує результат або помилку.
 */
async function predict(event) {
    event.preventDefault();
    const formData = new FormData(document.getElementById("predict-form"));
    const values = {};
    featureColumns.forEach(f => values[f] = formData.get(f));

    try {
        const response = await fetch("/basic_mode/predict-value", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({"values": values})
        });

        const result = await response.json();

        if ("prediction" in result) {
            let output = "<strong>Результат прогнозу:</strong> " + result.prediction;

            if (result.probabilities) {
                output += "<br><strong>Ймовірності класів:</strong><ul>";
                result.probabilities.forEach((p, i) => {
                    output += `<li>Клас ${i}: ${p.toFixed(3)}</li>`;
                });
                output += "</ul>";
            }

            if (result.cluster_size) {
                output += "<br><strong>Розмір кластера:</strong> " + result.cluster_size;
            }
            if (result.cluster_centroid) {
                output += "<br><strong>Центроїд кластера:</strong><ul>";
                result.cluster_centroid.forEach((val, i) => {
                    output += `<li>Ознака ${i}: ${val.toFixed(3)}</li>`;
                });
                output += "</ul>";
            }

            document.getElementById("prediction-result").innerHTML = output;
            logger("info", `Прогноз виконано: ${result.prediction}`)
        } else if (result.error) {

            logger("error", `Помилка при кластеризації: ${result.error}`)
            alert(result.error);
        } else {
            document.getElementById("prediction-result").innerHTML =
                "<strong>Немає результату прогнозу</strong>";
            logger("warn", "Прогноз не повернув результат")
        }
    } catch (e) {
        logger("error", `Помилка при запиті прогнозу: ${e.message}`)
    }
}

window.selectColumn = selectColumn;
window.trainModel = trainModel;
window.trainModelNoTarget = trainModelNoTarget;
window.predict = predict;