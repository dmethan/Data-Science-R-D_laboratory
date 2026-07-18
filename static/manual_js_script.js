// Єдиний словник параметрів для всіх моделей
import { logger } from "./logger.js";

// Функція для побудови графіка
async function generatePlot() {
    const column = document.getElementById("plot-column").value;
    const plotType = document.getElementById("plot-type").value;

    if (!column || !plotType) {
        alert("Оберіть колонку та тип графіка!");
        return;
    }

    try {
        const response = await fetch(`/manual_mode/analysis/plot?column=${column}&type=${plotType}`);
        if (!response.ok) throw new Error("Помилка при отриманні графіка");

        const data = await response.json();

        // Вставляємо графік у контейнер
        const plotContainer = document.getElementById("custom-plot-container");
        plotContainer.innerHTML = `
            <h4>${plotType} для ${column}</h4>
            <img src="data:image/png;base64,${data.img}" alt="Графік ${column}">
            <a download="${column}_${plotType}.png" href="data:image/png;base64,${data.img}">⬇️ Завантажити PNG</a>
        `;
    } catch (error) {
        console.error(error);
        alert("Не вдалося побудувати графік");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const buttons = document.querySelectorAll(".collapsible-btn");
    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            const content = btn.nextElementSibling;
            if (content.style.display === "block") {
                content.style.display = "none";
                btn.innerHTML = btn.innerHTML.replace("▲", "▼");
            } else {
                content.style.display = "block";
                btn.innerHTML = btn.innerHTML.replace("▼", "▲");
            }
        });
    });
});

async function applyMissingMethods() {
    const selects = document.querySelectorAll("select[name^='method_']");
    const methods = {};

    selects.forEach(sel => {
        const colName = sel.name.replace("method_", "");
        methods[colName] = sel.value;
    });

    try {
        const response = await fetch("/manual_mode/apply_missing", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(methods)
        });

        if (!response.ok) throw new Error("Помилка застосування методів");
        const result = await response.json();

        // Оновлюємо таблицю прев’ю CSV
        const tableContainer = document.querySelector(".table-container");
        let html = "<table><thead><tr>";
        result.columns.forEach(col => {
            html += `<th>${col}</th>`;
        });
        html += "</tr></thead><tbody>";

        result.table.forEach(row => {
            html += "<tr>";
            result.columns.forEach(col => {
                html += `<td>${row[col]}</td>`;
            });
            html += "</tr>";
        });

        html += "</tbody></table>";
        tableContainer.innerHTML = html;

        alert("Методи обробки пропусків застосовано успішно!");
    } catch (error) {
        console.error(error);
        alert("Не вдалося застосувати методи");
    }
}

async function applyEncodingMethods() {
    logger("info", `Викликано функцію applyEncodingMethods()`);
    const selects = document.querySelectorAll("select[name^='method_']");
    const methods = {};

    selects.forEach(sel => {
        const colName = sel.name.replace("method_", "");
        methods[colName] = sel.value;
    });

    try {
        const response = await fetch("/manual_mode/apply_encoding", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(methods)
        });

        if (!response.ok) throw new Error("Помилка застосування методів");
        const result = await response.json();

        // Оновлюємо прев’ю таблиці
        const tableContainer = document.querySelector(".table-container");
        let html = "<table><thead><tr>";
        result.columns.forEach(col => html += `<th>${col}</th>`);
        html += "</tr></thead><tbody>";
        result.table.forEach(row => {
            html += "<tr>";
            result.columns.forEach(col => html += `<td>${row[col]}</td>`);
            html += "</tr>";
        });
        html += "</tbody></table>";
        tableContainer.innerHTML = html;

        alert("Методи кодування/масштабування застосовано успішно!");
    } catch (error) {
        console.error(error);
        alert("Не вдалося застосувати методи");
    }
}

async function generateCorrelationMatrix() {
    try {
        const response = await fetch("/manual_mode/correlation_matrix");
        if (!response.ok) throw new Error("Помилка побудови матриці");
        const result = await response.json();

        const container = document.getElementById("corr-matrix-container");
        container.innerHTML = `
            <img src="data:image/png;base64,${result.img}" alt="Heatmap кореляційної матриці">
            <a download="correlation_matrix.png" href="data:image/png;base64,${result.img}">⬇️ Завантажити PNG</a>
        `;
    } catch (error) {
        console.error(error);
        alert("Не вдалося побудувати кореляційну матрицю");
    }
}

// Завантаження колонок і побудова чекбоксів
async function loadCorrCheckboxes() {
    try {
        const response = await fetch("/manual_mode/get_columns");
        const result = await response.json();

        const container = document.getElementById("corr-checkboxes");
        container.innerHTML = "";

        result.columns.forEach(col => {
            const checkbox = document.createElement("label");
            checkbox.className = "checkbox-label";
            checkbox.innerHTML = `
                <input type="checkbox" value="${col}"> ${col}
            `;
            container.appendChild(checkbox);
        });

        // додаємо кнопку для побудови матриці
        const buildBtnContainer = document.getElementById("corr-build-btn");
        buildBtnContainer.innerHTML = `
            <button class="btn-blue" onclick="generateSelectedCorrelation()">Побудувати матрицю для вибраних ознак</button>
        `;
    } catch (error) {
        console.error(error);
        alert("Не вдалося завантажити список колонок");
    }
}

// Побудова матриці для вибраних ознак
async function generateSelectedCorrelation() {
    const selected = Array.from(document.querySelectorAll("#corr-checkboxes input:checked"))
                          .map(cb => cb.value);

    if (selected.length === 0) {
        alert("Виберіть хоча б одну ознаку");
        return;
    }

    try {
        const response = await fetch("/manual_mode/correlation_selected", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(selected)
        });

        const result = await response.json();

        const container = document.getElementById("corr-results");

        // таблиця з відсортованими значеннями
        let tableHtml = "";
        result.selected.forEach(col => {
            tableHtml += `<h3>Кореляції з ${col}</h3>`;
            tableHtml += "<table class='corr-table'><tr><th>Ознака</th><th>Кореляція</th></tr>";
            result.correlations[col].forEach(([feature, value]) => {
                tableHtml += `<tr><td>${feature}</td><td>${value.toFixed(3)}</td></tr>`;
            });
            tableHtml += "</table><br>";
        });


        // тепловая карта
        const heatmapHtml = `
            <img src="data:image/png;base64,${result.img}" alt="Heatmap кореляцій">
            <a download="correlation_selected.png" href="data:image/png;base64,${result.img}">⬇️ Завантажити PNG</a>
        `;

        container.innerHTML = tableHtml + "<br>" + heatmapHtml;
    } catch (error) {
        console.error(error);
        alert("Не вдалося побудувати кореляційну матрицю");
    }
}


function showXYSelectors() {
    document.getElementById("xy-selectors").style.display = "block";
}

async function applyXYSelection() {
    const xValues = Array.from(document.querySelectorAll("input[name='x-columns']:checked")).map(cb => cb.value);
    const yValues = Array.from(document.querySelectorAll("input[name='y-columns']:checked")).map(cb => cb.value);

    const filteredY = yValues.filter(val => !xValues.includes(val));

    try {
        const response = await fetch("/manual_mode/apply_xy", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ X: xValues, Y: filteredY })
        });

        if (!response.ok) throw new Error("Помилка застосування X та Y");
        const result = await response.json();

        // Прев’ю X
        const xContainer = document.getElementById("x-preview");
        let xHtml = "<table><thead><tr>";
        result.X.forEach(col => xHtml += `<th>${col}</th>`);
        xHtml += "</tr></thead><tbody>";
        result.table.forEach(row => {
            xHtml += "<tr>";
            result.X.forEach(col => xHtml += `<td>${row[col]}</td>`);
            xHtml += "</tr>";
        });
        xHtml += "</tbody></table>";
        xContainer.innerHTML = xHtml;

        // Прев’ю Y
        const yContainer = document.getElementById("y-preview");
        let yHtml = "<table><thead><tr>";
        result.Y.forEach(col => yHtml += `<th>${col}</th>`);
        yHtml += "</tr></thead><tbody>";
        result.table.forEach(row => {
            yHtml += "<tr>";
            result.Y.forEach(col => yHtml += `<td>${row[col]}</td>`);
            yHtml += "</tr>";
        });
        yHtml += "</tbody></table>";
        yContainer.innerHTML = yHtml;

        alert("Вибір X та Y застосовано успішно!");
    } catch (error) {
        console.error(error);
        alert("Не вдалося застосувати X та Y");
    }
}

async function applyTrainTestSplit() {
    const testSize = parseFloat(document.getElementById("test-size").value);
    const valSize = parseFloat(document.getElementById("val-size").value);
    const randomState = parseInt(document.getElementById("random-state").value);

    try {
        const response = await fetch("/manual_mode/train_test_split", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ test_size: testSize, val_size: valSize, random_state: randomState })
        });

        if (!response.ok) throw new Error("Помилка розділення вибірки");
        const result = await response.json();

        // Вивід shape
        const shapesDiv = document.getElementById("split-shapes");
        shapesDiv.innerHTML = `
            📊 Shape вибірок:<br>
            X_train = ${result.shapes.x_train[0]} × ${result.shapes.x_train[1]}<br>
            X_val = ${result.shapes.x_val[0]} × ${result.shapes.x_val[1]}<br>
            X_test = ${result.shapes.x_test[0]} × ${result.shapes.x_test[1]}<br>
            Y_train = ${result.shapes.y_train[0]} × ${result.shapes.y_train[1]}<br>
            Y_val = ${result.shapes.y_val[0]} × ${result.shapes.y_val[1]}<br>
            Y_test = ${result.shapes.y_test[0]} × ${result.shapes.y_test[1]}
        `;

        // Прев’ю таблиць
        renderTable("x-train-preview", result.X_columns, result.x_train);
        renderTable("x-val-preview", result.X_columns, result.x_val);
        renderTable("x-test-preview", result.X_columns, result.x_test);
        renderTable("y-train-preview", result.Y_columns, result.y_train);
        renderTable("y-val-preview", result.Y_columns, result.y_val);
        renderTable("y-test-preview", result.Y_columns, result.y_test);

        alert("Розділення виконано успішно!");
    } catch (error) {
        console.error(error);
        alert("Не вдалося виконати train/test/validation split");
    }
}

function renderTable(containerId, columns, rows) {
    const container = document.getElementById(containerId);
    let html = "<table><thead><tr>";
    columns.forEach(col => html += `<th>${col}</th>`);
    html += "</tr></thead><tbody>";
    rows.forEach(row => {
        html += "<tr>";
        columns.forEach(col => html += `<td>${row[col]}</td>`);
        html += "</tr>";
    });
    html += "</tbody></table>";
    container.innerHTML = html;
}

async function refreshTransformations() {
    const response = await fetch("/manual_mode/transformations");
    const result = await response.json();

    const container = document.getElementById("transformations-table");
    let html = "<table><thead><tr><th>Час</th><th>Дія</th><th>Колонка</th><th>Метод</th></tr></thead><tbody>";
    result.history.forEach(row => {
        html += `<tr>
                    <td>${row.timestamp}</td>
                    <td>${row.action}</td>
                    <td>${row.column || "-"}</td>
                    <td>${row.method || "-"}</td>
                 </tr>`;
    });
    html += "</tbody></table>";
    container.innerHTML = html;
}

function loadTaskButtons() {
    // Отримуємо список цільових ознак Y з бекенду
    fetch("/manual_mode/get_y_info")
        .then(resp => resp.json())
        .then(data => {
            const Y = data.Y || [];
            const Y_dtypes = data.Y_dtypes || [];
            renderTaskButtons(Y, Y_dtypes);
        })
        .catch(err => {
            console.error(err);
            alert("Не вдалося отримати інформацію про Y");
        });
}

function renderTaskButtons(Y, Y_dtypes) {
    const container = document.getElementById("task-buttons");
    container.innerHTML = "";

    if (Y.length === 0) {
        container.innerHTML = `
            <button class="btn-blue" onclick="runNoTargetClusteringModels()">Кластеризація</button>
            <p>🔹 Кластеризація — групування об'єктів без цільової ознаки.</p>
        `;
    } else if (Y.length === 1) {
        container.innerHTML = `
            <button class="btn-blue" onclick="runRegressionModels()">Регресія</button>
            <p>📈 Регресія — прогноз числового значення.</p>

            <button class="btn-green" onclick="runClassificationModels()">Класифікація</button>
            <p>🔖 Класифікація — прогноз категорії.</p>

            <button class="btn-orange" onclick="runClusteringModels()">Кластеризація</button>
            <p>🔹 Кластеризація — групування об'єктів, можна порівняти з цільовою ознакою.</p>
        `;
    } else {
        const hasMixedTypes = new Set(Y_dtypes).size > 1;
        if (hasMixedTypes) {
            container.innerHTML = `
                <button class="btn-purple" onclick="loadTargetFeaturesForHybrid()">Змішаний прогноз ⚡</button>
                <p>⚡ Змішаний прогноз — одночасне передбачення числових та категоріальних ознак.</p>
            `;
        } else {
            container.innerHTML = `
                <button class="btn-blue" onclick="runMultiRegression()">Multi‑output Регресія</button>
                <p>📊 Multi‑output регресія — прогноз кількох числових ознак одночасно.</p>

                <button class="btn-green" onclick="runMultiClassification()">Multi‑output Класифікація</button>
                <p>🔖 Multi‑output класифікація — прогноз кількох категоріальних ознак одночасно.</p>
            `;
        }
    }
}


// 🔹 Функція для регресії
// 🔹 Функція для регресії
async function runRegressionModels() {
    const loadingDiv = document.getElementById("loading-animation");
    const loadingText = document.getElementById("loading-text");
    const resultsDiv = document.getElementById("regression-results");

    resultsDiv.innerHTML = "";
    loadingDiv.style.display = "block";
    loadingText.innerText = "🔄 Відбувається тренування моделей...";

    const messages = [
        "🔄 Відбувається тренування моделей...",
        "⚙️ Підбір гіперпараметрів...",
        "📊 Порівняння результатів...",
        "✅ Формування топ‑3 конфігурацій..."
    ];
    let i = 0;
    const interval = setInterval(() => {
        loadingText.innerText = messages[i % messages.length];
        i++;
    }, 2500);

    try {
        const response = await fetch("/manual_mode/run_regression_random", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        if (!response.ok) throw new Error("Помилка запуску регресії");
        const result = await response.json();

        clearInterval(interval);
        loadingDiv.style.display = "none";

        result.top_results.forEach(model => {
            const card = document.createElement("div");
            card.className = "regression-card";

            const safeValue = v => (v !== null && v !== undefined ? v.toFixed(3) : "-");

            // Метрики для валідації
            let valHtml = `
                <div class="metrics-column">
                    <h4>Валідаційна вибірка</h4>
                    <div class="regression-metrics">
                        <div class="metric-container"><strong>R²:</strong> ${safeValue(model.val_metrics.r2)}</div>
                        <div class="metric-container"><strong>MAE:</strong> ${safeValue(model.val_metrics.mae)}</div>
                        <div class="metric-container"><strong>RMSE:</strong> ${safeValue(model.val_metrics.rmse)}</div>
                        <div class="metric-container"><strong>Explained Variance:</strong> ${safeValue(model.val_metrics.explained_variance)}</div>
                    </div>
                </div>
            `;

            // Метрики для тесту (якщо є)
            let testHtml = "";
            if (model.test_metrics) {
                testHtml = `
                    <div class="metrics-column">
                        <h4>Тестова вибірка</h4>
                        <div class="regression-metrics">
                            <div class="metric-container"><strong>R²:</strong> ${safeValue(model.test_metrics.r2)}</div>
                            <div class="metric-container"><strong>MAE:</strong> ${safeValue(model.test_metrics.mae)}</div>
                            <div class="metric-container"><strong>RMSE:</strong> ${safeValue(model.test_metrics.rmse)}</div>
                            <div class="metric-container"><strong>Explained Variance:</strong> ${safeValue(model.test_metrics.explained_variance)}</div>
                        </div>
                    </div>
                `;
            }

            // Об’єднання у дві колонки
            let metricsBlock = `
                <div class="metrics-row">
                    ${valHtml}
                    ${testHtml}
                </div>
            `;


            // Найкращі параметри
            let paramsHtml = `
                <div class="metric-container">
                    <strong>Найкращі параметри:</strong> ${JSON.stringify(model.best_params)}
                </div>
            `;

            // Графіки
            let plotsHtml = "";
            if (model.plots) {
                Object.keys(model.plots).forEach(key => {
                    plotsHtml += `
                        <div class="plot-container">
                            <h4>${key}</h4>
                            <img src="data:image/png;base64,${model.plots[key]}" alt="${key}">
                        </div>
                    `;
                });
            }

            card.innerHTML = `
                <div class="model-header">
                    <h3>${model.model}</h3>
                </div>
                ${metricsBlock}
                ${paramsHtml}
                ${plotsHtml}
            `;
            resultsDiv.appendChild(card);
        });

    } catch (error) {
        clearInterval(interval);
        loadingDiv.style.display = "none";
        console.error(error);
        alert("Не вдалося запустити регресійні моделі");
    }
}



// 🔹 Функція для класифікації
async function runClassificationModels() {
    const loadingDiv = document.getElementById("loading-animation");
    const loadingText = document.getElementById("loading-text");
    const resultsDiv = document.getElementById("classification-results");

    resultsDiv.innerHTML = "";
    loadingDiv.style.display = "block";
    loadingText.innerText = "🔄 Відбувається тренування моделей...";

    const messages = [
        "🔄 Відбувається тренування моделей...",
        "⚙️ Підбір гіперпараметрів...",
        "📊 Порівняння результатів...",
        "✅ Формування топ‑3 конфігурацій..."
    ];
    let i = 0;
    const interval = setInterval(() => {
        loadingText.innerText = messages[i % messages.length];
        i++;
    }, 2500);

    try {
        const response = await fetch("/manual_mode/run_classification_random", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        if (!response.ok) throw new Error("Помилка запуску класифікації");
        const result = await response.json();

        clearInterval(interval);
        loadingDiv.style.display = "none";

        result.top_results.forEach(model => {
            const card = document.createElement("div");
            card.className = "classification-card";

            const safeValue = v => (v !== null && v !== undefined ? v.toFixed(3) : "-");

            // Метрики для валідації
            let valHtml = `
                <div class="metrics-column">
                    <h4>Валідаційна вибірка</h4>
                    <div class="classification-metrics">
                        <div class="metric-container">
                            <div class="metric-name">Accuracy — Точність</div>
                            <div class="metric-value">${safeValue(model.val_metrics.accuracy)}</div>
                            <div class="metric-explain">Частка правильних прогнозів<br><span class="ranges">0–0.5 погано, 0.5–0.8 середньо, >0.8 добре</span></div>
                        </div>
                        <div class="metric-container">
                            <div class="metric-name">Precision — Точність позитивних</div>
                            <div class="metric-value">${safeValue(model.val_metrics.precision)}</div>
                            <div class="metric-explain">Точність позитивних прогнозів<br><span class="ranges">чим ближче до 1, тим краще</span></div>
                        </div>
                        <div class="metric-container">
                            <div class="metric-name">Recall — Повнота</div>
                            <div class="metric-value">${safeValue(model.val_metrics.recall)}</div>
                            <div class="metric-explain">Частка правильно знайдених позитивних<br><span class="ranges">0–0.5 низько, 0.5–0.8 середньо, >0.8 добре</span></div>
                        </div>
                        <div class="metric-container">
                            <div class="metric-name">F1-score — F1‑міра</div>
                            <div class="metric-value">${safeValue(model.val_metrics.f1)}</div>
                            <div class="metric-explain">Баланс precision/recall<br><span class="ranges">0–0.4 слабко, 0.4–0.7 середньо, >0.7 добре</span></div>
                        </div>
                        <div class="metric-container">
                            <div class="metric-name">Balanced Accuracy — Збалансована точність</div>
                            <div class="metric-value">${safeValue(model.val_metrics.balanced_accuracy)}</div>
                            <div class="metric-explain">Усереднена точність по класах<br><span class="ranges">0.5 випадково, >0.7 добре</span></div>
                        </div>
                        <div class="metric-container">
                            <div class="metric-name">MCC — Коефіцієнт Маттьюса</div>
                            <div class="metric-value">${safeValue(model.val_metrics.mcc)}</div>
                            <div class="metric-explain">Кореляція прогнозів і реальності<br><span class="ranges">−1 погано, 0 випадково, +1 ідеально</span></div>
                        </div>
                        <div class="metric-container">
                            <div class="metric-name">Kappa — Каппа Коена</div>
                            <div class="metric-value">${safeValue(model.val_metrics.kappa)}</div>
                            <div class="metric-explain">Узгодженість прогнозів<br><span class="ranges">0–0.4 слабка, 0.4–0.6 середня, >0.6 добра</span></div>
                        </div>
                    </div>
                </div>
            `;

            // Метрики для тесту (якщо є)
            let testHtml = "";
            if (model.test_metrics) {
                testHtml = `
                    <div class="metrics-column">
                        <h4>Тестова вибірка</h4>
                        <div class="classification-metrics">
                            <div class="metric-container">
                                <div class="metric-name">Accuracy — Точність</div>
                                <div class="metric-value">${safeValue(model.test_metrics.accuracy)}</div>
                                <div class="metric-explain">Частка правильних прогнозів<br><span class="ranges">0–0.5 погано, 0.5–0.8 середньо, >0.8 добре</span></div>
                            </div>
                            <div class="metric-container">
                                <div class="metric-name">Precision — Точність позитивних</div>
                                <div class="metric-value">${safeValue(model.test_metrics.precision)}</div>
                                <div class="metric-explain">Точність позитивних прогнозів<br><span class="ranges">чим ближче до 1, тим краще</span></div>
                            </div>
                            <div class="metric-container">
                                <div class="metric-name">Recall — Повнота</div>
                                <div class="metric-value">${safeValue(model.test_metrics.recall)}</div>
                                <div class="metric-explain">Частка правильно знайдених позитивних<br><span class="ranges">0–0.5 низько, 0.5–0.8 середньо, >0.8 добре</span></div>
                            </div>
                            <div class="metric-container">
                                <div class="metric-name">F1-score — F1‑міра</div>
                                <div class="metric-value">${safeValue(model.test_metrics.f1)}</div>
                                <div class="metric-explain">Баланс precision/recall<br><span class="ranges">0–0.4 слабко, 0.4–0.7 середньо, >0.7 добре</span></div>
                            </div>
                            <div class="metric-container">
                                <div class="metric-name">Balanced Accuracy — Збалансована точність</div>
                                <div class="metric-value">${safeValue(model.test_metrics.balanced_accuracy)}</div>
                                <div class="metric-explain">Усереднена точність по класах<br><span class="ranges">0.5 випадково, >0.7 добре</span></div>
                            </div>
                            <div class="metric-container">
                                <div class="metric-name">MCC — Коефіцієнт Маттьюса</div>
                                <div class="metric-value">${safeValue(model.test_metrics.mcc)}</div>
                                <div class="metric-explain">Кореляція прогнозів і реальності<br><span class="ranges">−1 погано, 0 випадково, +1 ідеально</span></div>
                            </div>
                            <div class="metric-container">
                                <div class="metric-name">Kappa — Каппа Коена</div>
                                <div class="metric-value">${safeValue(model.test_metrics.kappa)}</div>
                                <div class="metric-explain">Узгодженість прогнозів<br><span class="ranges">0–0.4 слабка, 0.4–0.6 середня, >0.6 добра</span></div>
                            </div>
                        </div>
                    </div>
                `;
            }

            // Об’єднання у дві колонки
            let metricsBlock = `
                <div class="metrics-row">
                    ${valHtml}
                    ${testHtml}
                </div>
                <div class="metric-container">
                    <strong>Найкращі параметри:</strong> ${JSON.stringify(model.best_params)}
                </div>
            `;


            // Графіки
            let plotsHtml = "";
            if (model.plots) {
                Object.keys(model.plots).forEach(key => {
                    plotsHtml += `
                        <div class="plot-container">
                            <h4>${key}</h4>
                            <img src="data:image/png;base64,${model.plots[key]}" alt="${key}">
                        </div>
                    `;
                });
            }

            card.innerHTML = `
                <div class="model-header">
                    <h3>${model.model}</h3>
                </div>
                ${metricsBlock}
                ${plotsHtml}
            `;
            resultsDiv.appendChild(card);
        });

    } catch (error) {
        clearInterval(interval);
        loadingDiv.style.display = "none";
        console.error(error);
        alert("Не вдалося запустити класифікаційні моделі");
    }
}


async function runClusteringModels() {
    const loadingDiv = document.getElementById("loading-animation");
    const loadingText = document.getElementById("loading-text");
    const resultsDiv = document.getElementById("clustering-results");

    resultsDiv.innerHTML = "";
    loadingDiv.style.display = "block";
    loadingText.innerText = "🔄 Виконується кластеризація...";

    const messages = [
        "🔄 Виконується кластеризація...",
        "⚙️ Підбір параметрів...",
        "📊 Порівняння алгоритмів...",
        "✅ Формування топ‑результатів..."
    ];
    let i = 0;
    const interval = setInterval(() => {
        loadingText.innerText = messages[i % messages.length];
        i++;
    }, 2500);

    try {
        const response = await fetch("/manual_mode/run_clustering", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        if (!response.ok) throw new Error("Помилка запуску кластеризації");
        const result = await response.json();

        clearInterval(interval);
        loadingDiv.style.display = "none";

        result.top_results.forEach(model => {
            const card = document.createElement("div");
            card.className = "clustering-card";

            const safeValue = v => (v !== null && v !== undefined ? v.toFixed(3) : "-");

            // Метрики з поясненнями
            let metricsHtml = `
                <div class="clustering-metrics">
                    <div class="metric-container">
                        <div class="metric-name">ARI — Скоригований індекс Ранда</div>
                        <div class="metric-value">${safeValue(model.ari)}</div>
                        <div class="metric-explain">Вимірює схожість кластеризації з істинними класами<br><span class="ranges">−1 погано, 0 випадково, +1 ідеально</span></div>
                    </div>
                    <div class="metric-container">
                        <div class="metric-name">NMI — Нормалізована взаємна інформація</div>
                        <div class="metric-value">${safeValue(model.nmi)}</div>
                        <div class="metric-explain">Вимірює інформаційну схожість кластерів<br><span class="ranges">0 немає зв’язку, 1 повна відповідність</span></div>
                    </div>
                    <div class="metric-container">
                        <div class="metric-name">Homogeneity — Однорідність</div>
                        <div class="metric-value">${safeValue(model.homogeneity)}</div>
                        <div class="metric-explain">Наскільки кластери містять лише один клас<br><span class="ranges">0 низько, 1 ідеально</span></div>
                    </div>
                    <div class="metric-container">
                        <div class="metric-name">Silhouette — Силует</div>
                        <div class="metric-value">${safeValue(model.silhouette)}</div>
                        <div class="metric-explain">Якість кластеризації<br><span class="ranges">−1 погано, 0 випадково, +1 добре</span></div>
                    </div>
                    <div class="metric-container">
                        <div class="metric-name">Calinski-Harabasz — Індекс Калінскі‑Харабаса</div>
                        <div class="metric-value">${safeValue(model.calinski_harabasz)}</div>
                        <div class="metric-explain">Чим більше значення, тим краще розділені кластери</div>
                    </div>
                    <div class="metric-container">
                        <div class="metric-name">Davies-Bouldin — Індекс Девіса‑Болдіна</div>
                        <div class="metric-value">${safeValue(model.davies_bouldin)}</div>
                        <div class="metric-explain">Чим менше значення, тим краще</div>
                    </div>
                </div>
            `;

            // Розподіл кластерів
            let distributionHtml = "";
            let warningHtml = "";
            if (model.cluster_distribution) {
                const totalPoints = Object.values(model.cluster_distribution)
                    .reduce((sum, val) => sum + val, 0);

                Object.entries(model.cluster_distribution).forEach(([cid, count]) => {
                    distributionHtml += `кластер ${cid}: ${count} точок<br>`;
                    if (count / totalPoints > 0.7) {
                        warningHtml = `<div class="warning">⚠️ Кластер ${cid} містить ${count} точок (${((count/totalPoints)*100).toFixed(1)}%) — можливий дисбаланс</div>`;
                    }
                });
            } else {
                distributionHtml = "-";
            }

            // Графіки
            let plotsHtml = "";
            if (model.plots) {
                Object.keys(model.plots).forEach(key => {
                    if (model.plots[key]) {
                        plotsHtml += `
                            <div class="plot-container">
                                <h4>${key}</h4>
                                <img src="data:image/png;base64,${model.plots[key]}" alt="${key}">
                            </div>
                        `;
                    }
                });
            }

            card.innerHTML = `
                <h3>${model.model}</h3>
                ${metricsHtml}
                <div class="clustering-distribution">
                    <strong>Розподіл кластерів:</strong><br>${distributionHtml}
                    ${warningHtml}
                </div>
                ${plotsHtml}
            `;
            resultsDiv.appendChild(card);
        });

    } catch (error) {
        clearInterval(interval);
        loadingDiv.style.display = "none";
        console.error(error);
        alert("Не вдалося запустити кластеризацію");
    }
}


async function runNoTargetClusteringModels() {
    const loadingDiv = document.getElementById("loading-animation");
    const loadingText = document.getElementById("loading-text");
    const resultsDiv = document.getElementById("no-target-clustering-results");

    resultsDiv.innerHTML = "";
    loadingDiv.style.display = "block";
    loadingText.innerText = "🔄 Виконується кластеризація без цільової змінної...";

    const messages = [
        "🔄 Виконується кластеризація...",
        "⚙️ Підбір параметрів...",
        "📊 Обчислення метрик...",
        "✅ Формування результатів..."
    ];
    let i = 0;
    const interval = setInterval(() => {
        loadingText.innerText = messages[i % messages.length];
        i++;
    }, 2500);

    try {
        const response = await fetch("/manual_mode/run_no_target_clustering", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        if (!response.ok) throw new Error("Помилка запуску кластеризації");
        const result = await response.json();

        clearInterval(interval);
        loadingDiv.style.display = "none";

        result.results.forEach(model => {
        const card = document.createElement("div");
        card.className = "clustering-card";

        const safeValue = v => (v !== null && v !== undefined ? v.toFixed(3) : "-");

        // Метрики з поясненнями
        let metricsHtml = `
            <div class="clustering-metrics">
                <div class="metric-container">
                    <div class="metric-name">Train Silhouette — Силует</div>
                    <div class="metric-value">${safeValue(model.train_silhouette)}</div>
                    <div class="metric-explain">Якість кластеризації<br><span class="ranges">−1 погано, 0 випадково, +1 добре</span></div>
                </div>
                <div class="metric-container">
                    <div class="metric-name">Train Calinski-Harabasz — Індекс Калінскі‑Харабаса</div>
                    <div class="metric-value">${safeValue(model.train_calinski_harabasz)}</div>
                    <div class="metric-explain">Чим більше значення, тим краще розділені кластери</div>
                </div>
                <div class="metric-container">
                    <div class="metric-name">Train Davies-Bouldin — Індекс Девіса‑Болдіна</div>
                    <div class="metric-value">${safeValue(model.train_davies_bouldin)}</div>
                    <div class="metric-explain">Чим менше значення, тим краще</div>
                </div>
                <div class="metric-container">
                    <div class="metric-name">Test Silhouette — Силует</div>
                    <div class="metric-value">${safeValue(model.test_silhouette)}</div>
                    <div class="metric-explain">Якість кластеризації<br><span class="ranges">−1 погано, 0 випадково, +1 добре</span></div>
                </div>
                <div class="metric-container">
                    <div class="metric-name">Test Calinski-Harabasz — Індекс Калінскі‑Харабаса</div>
                    <div class="metric-value">${safeValue(model.test_calinski_harabasz)}</div>
                    <div class="metric-explain">Чим більше значення, тим краще розділені кластери</div>
                </div>
                <div class="metric-container">
                    <div class="metric-name">Test Davies-Bouldin — Індекс Девіса‑Болдіна</div>
                    <div class="metric-value">${safeValue(model.test_davies_bouldin)}</div>
                    <div class="metric-explain">Чим менше значення, тим краще</div>
                </div>
            </div>
        `;

        // Графіки з словника plots
        let plotsHtml = "";
        if (model.plots) {
            Object.keys(model.plots).forEach(key => {
                if (model.plots[key]) {
                    plotsHtml += `
                        <div class="plot-container">
                            <h4>${key}</h4>
                            <img src="data:image/png;base64,${model.plots[key]}" alt="${key}">
                        </div>
                    `;
                }
            });
        }

        card.innerHTML = `
            <h3>${model.model}</h3>
            ${metricsHtml}
            ${plotsHtml}
        `;
        resultsDiv.appendChild(card);
    });


    } catch (error) {
        clearInterval(interval);
        loadingDiv.style.display = "none";
        console.error(error);
        alert("Не вдалося запустити кластеризацію без цільової змінної");
    }
}


async function runHybridPrediction() {
    const loadingDiv = document.getElementById("loading-animation");
    const loadingText = document.getElementById("loading-text");
    const resultsDiv = document.getElementById("hybrid-prediction-results");

    resultsDiv.innerHTML = "";
    loadingDiv.style.display = "block";
    loadingText.innerText = "🔄 Виконується гібридне прогнозування...";

    const messages = [
        "🔄 Виконується гібридне прогнозування...",
        "⚙️ Навчання моделей...",
        "📊 Обчислення результатів...",
        "✅ Формування прогнозів..."
    ];
    let i = 0;
    const interval = setInterval(() => {
        loadingText.innerText = messages[i % messages.length];
        i++;
    }, 2500);

    try {
        const numericTargets = [];
        const categoricalTargets = [];
        document.querySelectorAll("#numeric-targets input:checked").forEach(cb => numericTargets.push(cb.value));
        document.querySelectorAll("#categorical-targets input:checked").forEach(cb => categoricalTargets.push(cb.value));

        logger("info", `Вибрані числові ознаки: ${numericTargets.join(", ")}`);
        logger("info", `Вибрані категоріальні ознаки: ${categoricalTargets.join(", ")}`);

        const payload = { numeric_targets: numericTargets, categorical_targets: categoricalTargets };
        logger("info", `Формуємо запит до бекенду: ${JSON.stringify(payload)}`);

        const response = await fetch("/manual_mode/run_hybrid_prediction", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Помилка запуску гібридного прогнозування");
        const result = await response.json();

        clearInterval(interval);
        loadingDiv.style.display = "none";

        logger("info", "Отримано результати від бекенду");

        // Формування таблиці результатів
        let tableLog = "\n=== РЕЗУЛЬТАТИ ГІБРИДНОГО ПРОГНОЗУ ===\n";

        Object.keys(result).forEach(section => {
            if (Array.isArray(result[section]) && result[section].length > 0) {
                tableLog += `\n--- ${section.toUpperCase()} ---\n`;

                result[section].forEach(model => {
                    tableLog += `Ознака: ${model.target}\n`;

                    // Вивід топ моделей
                    if (model.top_models) {
                        model.top_models.forEach((m, idx) => {
                            tableLog += `  Модель #${idx+1}: ${m.model}\n`;
                            tableLog += `    Параметри: ${JSON.stringify(m.params)}\n`;

                            if (section === "classification") {
                                tableLog += `    Accuracy: ${m.metrics.accuracy.toFixed(3)}, F1: ${m.metrics.f1_score.toFixed(3)}, Precision: ${m.metrics.precision.toFixed(3)}, Recall: ${m.metrics.recall.toFixed(3)}\n`;
                                tableLog += `    Balanced Accuracy: ${m.metrics.balanced_accuracy?.toFixed(3) || "N/A"}, MCC: ${m.metrics.mcc?.toFixed(3) || "N/A"}, Kappa: ${m.metrics.kappa?.toFixed(3) || "N/A"}, Log-loss: ${m.metrics.logloss ? m.metrics.logloss.toFixed(3) : "N/A"}\n`;
                            } else if (section === "regression") {
                                tableLog += `    R²: ${m.metrics.r2_score.toFixed(3)}, Adjusted R²: ${m.metrics.adjusted_r2?.toFixed(3) || "N/A"}, MAE: ${m.metrics.mae.toFixed(3)}, RMSE: ${m.metrics.rmse.toFixed(3)}\n`;
                                tableLog += `    Median AE: ${m.metrics.median_ae?.toFixed(3) || "N/A"}, MSLE: ${m.metrics.msle?.toFixed(3) || "N/A"}, Explained Variance: ${m.metrics.explained_variance?.toFixed(3) || "N/A"}\n`;
                            }
                        });
                    }

                    // Вивід графіків (тільки перші 10 символів base64)
                    [
                        "confusion_matrix_plot","roc_plot","pr_curve","feature_importance_plot",
                        "scatter_plot","residual_plot","residuals_distribution","qq_plot"
                    ].forEach(plotKey => {
                        if (model[plotKey]) {
                            const preview = model[plotKey].substring(0,10);
                            tableLog += `  ${plotKey}: base64 початок -> ${preview}...\n`;
                        }
                    });
                });
            }
        });

        // Вивід у лог одним викликом
        logger("debug", tableLog);




        resultsDiv.innerHTML = "";

        // Класифікація
        if (result.classification && result.classification.length > 0) {
            result.classification.forEach(model => {
                const card = document.createElement("div");
                card.className = "hybrid-card";

                let modelsHtml = "";
                model.top_models.forEach(m => {
                    let graphsHtml = "";
                    if (m.confusion_matrix_plot) {
                        graphsHtml += `<img src="data:image/png;base64,${m.confusion_matrix_plot}" alt="Confusion Matrix"><br>`;
                    }
                    if (m.roc_plot) {
                        graphsHtml += `<img src="data:image/png;base64,${m.roc_plot}" alt="ROC Curve"><br>`;
                    }
                    if (m.pr_curve) {
                        graphsHtml += `<img src="data:image/png;base64,${m.pr_curve}" alt="Precision-Recall Curve"><br>`;
                    }
                    if (m.feature_importance_plot) {
                        graphsHtml += `<img src="data:image/png;base64,${m.feature_importance_plot}" alt="Feature Importance"><br>`;
                    }
                    if (m.learning_curve_plot) {
                        graphsHtml += `<img src="data:image/png;base64,${m.learning_curve_plot}" alt="Learning Curve"><br>`;
                    }
                    if (m.validation_curve_plot) {
                        graphsHtml += `<img src="data:image/png;base64,${m.validation_curve_plot}" alt="Validation Curve"><br>`;
                    }

                    modelsHtml += `
                       <div class="model-block">
                            <div class="model-header">
                                <div class="model-name">${m.model}</div>
                                <div class="model-params">Params: ${JSON.stringify(m.params)}</div>
                            </div>

                        
                            <div class="metric-container">
                                <div class="metric-name">Accuracy</div>
                                <div class="metric-value">${m.metrics.accuracy?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Частка правильних прогнозів<br><span class="ranges">0–0.5 погано, 0.5–0.8 середньо, >0.8 добре</span></div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">F1-score</div>
                                <div class="metric-value">${m.metrics.f1_score?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Баланс precision/recall<br><span class="ranges">0–0.4 слабко, 0.4–0.7 середньо, >0.7 добре</span></div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">Precision</div>
                                <div class="metric-value">${m.metrics.precision?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Точність позитивних прогнозів<br><span class="ranges">чим ближче до 1, тим краще</span></div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">Recall</div>
                                <div class="metric-value">${m.metrics.recall?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Повнота виявлення класів<br><span class="ranges">0–0.5 низько, 0.5–0.8 середньо, >0.8 добре</span></div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">Balanced Accuracy</div>
                                <div class="metric-value">${m.metrics.balanced_accuracy?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Усереднена точність по класах<br><span class="ranges">0.5 випадково, >0.7 добре</span></div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">MCC</div>
                                <div class="metric-value">${m.metrics.mcc?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Кореляція прогнозів і реальності<br><span class="ranges">−1 погано, 0 випадково, +1 ідеально</span></div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">Kappa</div>
                                <div class="metric-value">${m.metrics.kappa?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Узгодженість прогнозів<br><span class="ranges">0–0.4 слабка, 0.4–0.6 середня, >0.6 добра</span></div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">Log-loss</div>
                                <div class="metric-value">${m.metrics.logloss ? m.metrics.logloss.toFixed(3) : "N/A"}</div>
                                <div class="metric-explain">Якість ймовірнісних прогнозів<br><span class="ranges">чим ближче до 0, тим краще</span></div>
                            </div>
                        
                            <br>${graphsHtml}
                        </div>


                    `;
                });

                card.innerHTML = `<h3>Класифікація: ${model.target}</h3>${modelsHtml}`;
                resultsDiv.appendChild(card);
                logger("info", `Відображено класифікацію для ознаки ${model.target}`);
            });
        }


        // Регресія
        if (result.regression && result.regression.length > 0) {
            result.regression.forEach(model => {
                const card = document.createElement("div");
                card.className = "hybrid-card";

                let modelsHtml = "";
                model.top_models.forEach(m => {
                    let graphsHtml = "";
                    if (m.scatter_plot) {
                        graphsHtml += `<img src="data:image/png;base64,${m.scatter_plot}" alt="Scatter Plot"><br>`;
                    }
                    if (m.residual_plot) {
                        graphsHtml += `<img src="data:image/png;base64,${m.residual_plot}" alt="Residual Plot"><br>`;
                    }
                    if (m.residuals_distribution) {
                        graphsHtml += `<img src="data:image/png;base64,${m.residuals_distribution}" alt="Residuals Distribution"><br>`;
                    }
                    if (m.qq_plot) {
                        graphsHtml += `<img src="data:image/png;base64,${m.qq_plot}" alt="Q-Q Plot"><br>`;
                    }
                    if (m.feature_importance_plot) {
                        graphsHtml += `<img src="data:image/png;base64,${m.feature_importance_plot}" alt="Feature Importance"><br>`;
                    }
                    if (m.learning_curve_plot) {
                        graphsHtml += `<img src="data:image/png;base64,${m.learning_curve_plot}" alt="Learning Curve"><br>`;
                    }
                    if (m.validation_curve_plot) {
                        graphsHtml += `<img src="data:image/png;base64,${m.validation_curve_plot}" alt="Validation Curve"><br>`;
                    }

                    modelsHtml += `
                        <div class="model-block">
                            <div class="model-header">
                                <div class="model-name">${m.model}</div>
                                <div class="model-params">Params: ${JSON.stringify(m.params)}</div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">R²</div>
                                <div class="metric-value">${m.metrics.r2_score?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Частка поясненої дисперсії<br><span class="ranges">0 погано, 0.5 середньо, >0.7 добре</span></div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">Adjusted R²</div>
                                <div class="metric-value">${m.metrics.adjusted_r2?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Скоригований R² з урахуванням кількості ознак</div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">MAE</div>
                                <div class="metric-value">${m.metrics.mae?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Середня абсолютна похибка<br><span class="ranges">чим менше, тим краще</span></div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">RMSE</div>
                                <div class="metric-value">${m.metrics.rmse?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Корінь середньоквадратичної похибки<br><span class="ranges">0 ідеально, >значення цільової змінної — погано</span></div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">Median AE</div>
                                <div class="metric-value">${m.metrics.median_ae?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Медіанна похибка, стійка до викидів</div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">MSLE</div>
                                <div class="metric-value">${m.metrics.msle?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Логарифмічна похибка<br><span class="ranges">0 ідеально, <0.2 добре</span></div>
                            </div>
                        
                            <div class="metric-container">
                                <div class="metric-name">Explained Variance</div>
                                <div class="metric-value">${m.metrics.explained_variance?.toFixed(3) || ""}</div>
                                <div class="metric-explain">Частка дисперсії, поясненої моделлю<br><span class="ranges">0–0.5 слабко, >0.7 добре</span></div>
                            </div>
                        
                            <br>${graphsHtml}
                        </div>
                    `;
                });

                card.innerHTML = `<h3>Регресія: ${model.target}</h3>${modelsHtml}`;
                resultsDiv.appendChild(card);
                logger("info", `Відображено регресію для ознаки ${model.target}`);
            });
        }



    } catch (error) {
        clearInterval(interval);
        loadingDiv.style.display = "none";
        logger("error", `Помилка запуску гібридного прогнозування: ${error}`);
        alert("Не вдалося запустити гібридне прогнозування");
    }
}


async function runMultiRegression() {
    const loadingDiv = document.getElementById("loading-animation");
    const loadingText = document.getElementById("loading-text");
    const resultsDiv = document.getElementById("multi-regression-results");

    resultsDiv.innerHTML = "";
    loadingDiv.style.display = "block";
    loadingText.innerText = "🔄 Виконується мульти‑регресія...";

    const messages = [
        "🔄 Виконується мульти‑регресія...",
        "⚙️ Навчання моделей...",
        "📊 Порівняння результатів...",
        "✅ Формування топ‑результатів..."
    ];
    let i = 0;
    const interval = setInterval(() => {
        loadingText.innerText = messages[i % messages.length];
        i++;
    }, 2500);

    try {
        const response = await fetch("/manual_mode/run_multi_regression", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        if (!response.ok) throw new Error("Помилка запуску мульти‑регресії");
        const result = await response.json();

        clearInterval(interval);
        loadingDiv.style.display = "none";

        result.results.forEach(model => {
        const card = document.createElement("div");
        card.className = "regression-card";

        const safeValue = v => (v !== null && v !== undefined ? v.toFixed(3) : "-");

        // Метрики
        let metricsHtml = `
            <div class="regression-metrics">
        <div class="metric-container">
            <div class="metric-name">Train MSE — Середньоквадратична похибка (Train)</div>
            <div class="metric-value">${safeValue(model.train_mse)}</div>
            <div class="metric-explain">
                Середня квадратична похибка на тренуванні<br>
                <span class="ranges">чим менше, тим краще</span>
            </div>
        </div>
    
        <div class="metric-container">
            <div class="metric-name">Train R² — Коефіцієнт детермінації (Train)</div>
            <div class="metric-value">${safeValue(model.train_r2)}</div>
            <div class="metric-explain">
                Частка поясненої дисперсії<br>
                <span class="ranges">0 погано, 0.5 середньо, >0.7 добре</span>
            </div>
        </div>
    
        <div class="metric-container">
            <div class="metric-name">Val MSE — Середньоквадратична похибка (Val)</div>
            <div class="metric-value">${safeValue(model.val_mse)}</div>
            <div class="metric-explain">
                Похибка на валідації<br>
                <span class="ranges">чим менше, тим краще</span>
            </div>
        </div>
    
        <div class="metric-container">
            <div class="metric-name">Val R² — Коефіцієнт детермінації (Val)</div>
            <div class="metric-value">${safeValue(model.val_r2)}</div>
            <div class="metric-explain">
                Частка поясненої дисперсії на валідації<br>
                <span class="ranges">0 погано, 0.5 середньо, >0.7 добре</span>
            </div>
        </div>
    
        <div class="metric-container">
            <div class="metric-name">Test MSE — Середньоквадратична похибка (Test)</div>
            <div class="metric-value">${safeValue(model.test_mse)}</div>
            <div class="metric-explain">
                Похибка на тестових даних<br>
                <span class="ranges">чим менше, тим краще</span>
            </div>
        </div>
    
        <div class="metric-container">
            <div class="metric-name">Test R² — Коефіцієнт детермінації (Test)</div>
            <div class="metric-value">${safeValue(model.test_r2)}</div>
            <div class="metric-explain">
                Частка поясненої дисперсії на тесті<br>
                <span class="ranges">0 погано, 0.5 середньо, >0.7 добре</span>
            </div>
        </div>
    </div>

        `;

        // Графіки
        let plotsHtml = "";
        if (model.plots) {
            Object.keys(model.plots).forEach(key => {
                plotsHtml += `
                    <div class="plot-container">
                        <h4>${key.replace(/_/g," ")}</h4>
                        <img src="data:image/png;base64,${model.plots[key]}" alt="${key}">
                    </div>
                `;
            });
        }

        card.innerHTML = `
            <div class="model-header">
                <div class="model-name">${model.model}</div>
            </div>
            ${metricsHtml}
            ${plotsHtml}
        `;
        resultsDiv.appendChild(card);
    });


    } catch (error) {
        clearInterval(interval);
        loadingDiv.style.display = "none";
        console.error(error);
        alert("Не вдалося запустити мульти‑регресію");
    }
}


async function runMultiClassification() {
    const loadingDiv = document.getElementById("loading-animation");
    const loadingText = document.getElementById("loading-text");
    const resultsDiv = document.getElementById("multi-classification-results");

    resultsDiv.innerHTML = "";
    loadingDiv.style.display = "block";
    loadingText.innerText = "🔄 Виконується мульти‑класифікація...";

    const messages = [
        "🔄 Виконується мульти‑класифікація...",
        "⚙️ Навчання моделей...",
        "📊 Порівняння результатів...",
        "✅ Формування топ‑результатів..."
    ];
    let i = 0;
    const interval = setInterval(() => {
        loadingText.innerText = messages[i % messages.length];
        i++;
    }, 2500);

    try {
        const response = await fetch("/manual_mode/run_multi_classification", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        if (!response.ok) throw new Error("Помилка запуску мульти‑класифікації");
        const result = await response.json();

        clearInterval(interval);
        loadingDiv.style.display = "none";

        result.results.forEach(model => {
    const card = document.createElement("div");
    card.className = "classification-card";

    const safeValue = v => (v !== null && v !== undefined ? v.toFixed(3) : "-");

    // Метрики
    let metricsHtml = `
        <div class="classification-metrics">
            <div class="metric-container">
                <div class="metric-name">Train Accuracy — Точність (Train)</div>
                <div class="metric-value">${safeValue(model.train_accuracy)}</div>
                <div class="metric-explain">
                    Частка правильних прогнозів на тренуванні<br>
                    <span class="ranges">0–0.5 погано, 0.5–0.8 середньо, >0.8 добре</span>
                </div>
            </div>
    
            <div class="metric-container">
                <div class="metric-name">Train F1 — F1‑міра (Train)</div>
                <div class="metric-value">${safeValue(model.train_f1)}</div>
                <div class="metric-explain">
                    Баланс precision/recall на тренуванні<br>
                    <span class="ranges">0–0.4 слабко, 0.4–0.7 середньо, >0.7 добре</span>
                </div>
            </div>
    
            <div class="metric-container">
                <div class="metric-name">Val Accuracy — Точність (Val)</div>
                <div class="metric-value">${safeValue(model.val_accuracy)}</div>
                <div class="metric-explain">
                    Частка правильних прогнозів на валідації<br>
                    <span class="ranges">0–0.5 погано, 0.5–0.8 середньо, >0.8 добре</span>
                </div>
            </div>
    
            <div class="metric-container">
                <div class="metric-name">Val F1 — F1‑міра (Val)</div>
                <div class="metric-value">${safeValue(model.val_f1)}</div>
                <div class="metric-explain">
                    Баланс precision/recall на валідації<br>
                    <span class="ranges">0–0.4 слабко, 0.4–0.7 середньо, >0.7 добре</span>
                </div>
            </div>
    
            <div class="metric-container">
                <div class="metric-name">Test Accuracy — Точність (Test)</div>
                <div class="metric-value">${safeValue(model.test_accuracy)}</div>
                <div class="metric-explain">
                    Частка правильних прогнозів на тесті<br>
                    <span class="ranges">0–0.5 погано, 0.5–0.8 середньо, >0.8 добре</span>
                </div>
            </div>
    
            <div class="metric-container">
                <div class="metric-name">Test F1 — F1‑міра (Test)</div>
                <div class="metric-value">${safeValue(model.test_f1)}</div>
                <div class="metric-explain">
                    Баланс precision/recall на тесті<br>
                    <span class="ranges">0–0.4 слабко, 0.4–0.7 середньо, >0.7 добре</span>
                </div>
            </div>
        </div>
    `;


    // Графіки
    let plotsHtml = "";
    if (model.plots) {
        Object.keys(model.plots).forEach(sectionKey => {
            const sectionPlots = model.plots[sectionKey];
            sectionPlots.forEach(plotObj => {
                plotsHtml += `
                    <div class="plot-container">
                        <h4>${plotObj.title}</h4>
                        <img src="data:image/png;base64,${plotObj.plot}" alt="${plotObj.title}">
                    </div>
                `;
            });
        });
    }


    card.innerHTML = `
        <div class="model-header">
            <div class="model-name">${model.model}</div>
        </div>
        ${metricsHtml}
        ${plotsHtml}
    `;
    resultsDiv.appendChild(card);
});


    } catch (error) {
        clearInterval(interval);
        loadingDiv.style.display = "none";
        console.error(error);
        alert("Не вдалося запустити мульти‑класифікацію");
    }
}


async function loadTargetFeaturesForHybrid() {
    try {
        logger("info", "Виконується запит на отримання цільових ознак для змішаного прогнозу...");
        const response = await fetch("/manual_mode/get_target_features_for_hybrid");
        if (!response.ok) throw new Error("Не вдалося отримати цільові ознаки");

        const data = await response.json();
        const targetFeatures = data.target_features;

        logger("info", `Отримано ${targetFeatures.length} ознак: ${targetFeatures.join(", ")}`);

        // показуємо секцію з чекбоксами
        showHybridSelection(targetFeatures);

    } catch (error) {
        logger("error", `Помилка при завантаженні ознак Y: ${error}`);
        alert("Помилка при завантаженні ознак Y");
    }
}

function showHybridSelection(targetFeatures) {
    const hybridSection = document.getElementById("hybrid-selection");
    const numericDiv = document.getElementById("numeric-targets");
    const categoricalDiv = document.getElementById("categorical-targets");

    numericDiv.innerHTML = "";
    categoricalDiv.innerHTML = "";

    logger("info", "Формування чекбоксів для ознак...");

    targetFeatures.forEach(feature => {
        // числова ознака
        const numCheckbox = document.createElement("input");
        numCheckbox.type = "checkbox";
        numCheckbox.value = feature;
        numCheckbox.id = "num_" + feature;

        const numLabel = document.createElement("label");
        numLabel.htmlFor = numCheckbox.id;
        numLabel.innerText = feature;

        numericDiv.appendChild(numCheckbox);
        numericDiv.appendChild(numLabel);
        numericDiv.appendChild(document.createElement("br"));

        // категоріальна ознака
        const catCheckbox = document.createElement("input");
        catCheckbox.type = "checkbox";
        catCheckbox.value = feature;
        catCheckbox.id = "cat_" + feature;

        const catLabel = document.createElement("label");
        catLabel.htmlFor = catCheckbox.id;
        catLabel.innerText = feature;

        categoricalDiv.appendChild(catCheckbox);
        categoricalDiv.appendChild(catLabel);
        categoricalDiv.appendChild(document.createElement("br"));

        logger("info", `Додано ознаку '${feature}' у списки (num/cat)`);
    });

    hybridSection.style.display = "block";
    logger("info", "Секція для вибору ознак показана користувачу");
}


function openOverlay(imgSrc) {
  const overlay = document.getElementById("image-overlay");
  const overlayImg = document.getElementById("overlay-img");
  overlay.style.display = "block";
  overlayImg.src = imgSrc;
}


function closeOverlay() {
  document.getElementById("image-overlay").style.display = "none";
}


async function generateReport(predictionData) {
  const response = await fetch("/manual_mode/generate_report", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(predictionData)
  });

  if (!response.ok) {
    alert("❌ Помилка при створенні звіту");
    return;
  }

  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "report.pdf";
  a.click();
  window.URL.revokeObjectURL(url);
}

window.generatePlot = generatePlot;
window.applyMissingMethods = applyMissingMethods;
window.applyEncodingMethods = applyEncodingMethods;
window.generateCorrelationMatrix = generateCorrelationMatrix;
window.generateSelectedCorrelation = generateSelectedCorrelation;
window.loadCorrCheckboxes = loadCorrCheckboxes;
window.showXYSelectors = showXYSelectors;
window.applyXYSelection = applyXYSelection;
window.applyTrainTestSplit = applyTrainTestSplit;
window.refreshTransformations = refreshTransformations;
window.renderTaskButtons = renderTaskButtons;
window.runRegressionModels = runRegressionModels;
window.runClassificationModels = runClassificationModels;
window.runClusteringModels = runClusteringModels;
window.runNoTargetClusteringModels = runNoTargetClusteringModels;
window.runHybridPrediction = runHybridPrediction;
window.runMultiRegression = runMultiRegression;
window.runMultiClassification = runMultiClassification;
window.loadTargetFeaturesForHybrid = loadTargetFeaturesForHybrid;
window.showHybridSelection = showHybridSelection;
window.loadTaskButtons = loadTaskButtons;
window.openOverlay = openOverlay;
window.closeOverlay = closeOverlay;
window.generateReport = generateReport;
