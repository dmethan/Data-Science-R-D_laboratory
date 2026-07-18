// Єдиний словник параметрів для всіх моделей
import { logger } from "./logger.js";

const PARAMS_CONFIG = {
  // === Регресія ===
  "linear_regression": {
    "fit_intercept": {type: "bool", default: true},
    "positive": {type: "bool", default: false},
    "test_split": {type: "float", min: 0.1, max: 0.5, step: 0.1, default: 0.2},
    "random_state": {type: "int", default: 42}
  },
  "random_forest": {
    "n_estimators": {type: "int", min: 10, max: 1000, default: 100},
    "max_depth": {type: "int", min: 1, max: 50, default: null},
    "min_samples_split": {type: "int", min: 2, max: 50, default: 2},
    "min_samples_leaf": {type: "int", min: 1, max: 50, default: 1},
    "test_split": {type: "float", min: 0.1, max: 0.5, step: 0.1, default: 0.2},
    "random_state": {type: "int", default: 42}
  },
  "gradient_boosting": {
    "n_estimators": {type: "int", min: 10, max: 1000, default: 100},
    "learning_rate": {type: "float", min: 0.01, max: 1.0, step: 0.01, default: 0.1},
    "max_depth": {type: "int", min: 1, max: 20, default: 3},
    "test_split": {type: "float", min: 0.1, max: 0.5, step: 0.1, default: 0.2},
    "random_state": {type: "int", default: 42}
  },
  "xgboost": {
    "n_estimators": {type: "int", min: 10, max: 1000, default: 100},
    "learning_rate": {type: "float", min: 0.1, max: 0.9, step: 0.1, default: 0.3},
    "max_depth": {type: "int", min: 1, max: 20, default: 6},
    "subsample": {type: "float", min: 0.5, max: 1.0, step: 0.1, default: 1.0},
    "colsample_bytree": {type: "float", min: 0.5, max: 1.0, step: 0.1, default: 1.0},
    "objective": {type: "select", options: ["reg:squarederror","reg:logistic","reg:absoluteerror","reg:squaredlogerror","reg:pseudohubererror","reg:tweedie","reg:gamma"], default: "reg:squarederror"},
    "gamma": {type: "float", min: 0.0, max: 10.0, step: 0.1, default: 0.0},
    "test_split": {type: "float", min: 0.1, max: 0.5, step: 0.1, default: 0.2},
    "random_state": {type: "int", default: 0}
  },

  // === Класифікація ===
  "decision_tree": {
    "criterion": {type: "select", options: ["gini","entropy","log_loss"], default: "gini"},
    "max_depth": {type: "int", min: 1, max: 50, default: null},
    "test_split": {type: "float", min: 0.1, max: 0.5, step: 0.1, default: 0.2},
    "random_state": {type: "int", default: 42}
  },
  "gradient_boosting_cls": {
    "n_estimators": {type: "int", min: 10, max: 1000, default: 100},
    "learning_rate": {type: "float", min: 0.01, max: 1.0, step: 0.01, default: 0.1},
    "max_depth": {type: "int", min: 1, max: 20, default: 3},
    "test_split": {type: "float", min: 0.1, max: 0.5, step: 0.1, default: 0.2},
    "random_state": {type: "int", default: 42}
  },
  "knn": {
    "n_neighbors": {type: "int", min: 1, max: 50, default: 5},
    "test_split": {type: "float", min: 0.1, max: 0.5, step: 0.1, default: 0.2},
    "random_state": {type: "int", default: 42}
  },
  "logistic_regression": {
    "l1_ratio": {type: "float", min: 0.0, max: 1.0, step: 0.1, default: 0.5},
    "C": {type: "float", min: 0.0001, max: 1000000, step: 0.1, default: 1.0},
    "solver": {type: "select", options: ["lbfgs","liblinear","saga"], default: "lbfgs"},
    "max_iter": {type: "int", min: 50, max: 1000, default: 100},
    "test_split": {type: "float", min: 0.1, max: 0.5, step: 0.1, default: 0.2},
    "random_state": {type: "int", default: 42}
  },
  "random_forest_cls": {
    "n_estimators": {type: "int", min: 10, max: 1000, default: 100},
    "criterion": {type: "select", options: ["gini","entropy","log_loss"], default: "gini"},
    "max_depth": {type: "int", min: 1, max: 50, default: null},
    "test_split": {type: "float", min: 0.1, max: 0.5, step: 0.1, default: 0.2},
    "random_state": {type: "int", default: 42}
  },
  "svm": {
    "kernel": {type: "select", options: ["linear","poly","rbf","sigmoid"], default: "rbf"},
    "C": {type: "float", min: 0.01, max: 100.0, step: 0.1, default: 1.0},
    "test_split": {type: "float", min: 0.1, max: 0.5, step: 0.1, default: 0.2},
    "random_state": {type: "int", default: 42}
  },
  "xgboost_cls": {
    "n_estimators": {type: "int", min: 10, max: 1000, default: 100},
    "learning_rate": {type: "float", min: 0.01, max: 1.0, step: 0.01, default: 0.1},
    "max_depth": {type: "int", min: 1, max: 20, default: 6},
    "subsample": {type: "float", min: 0.5, max: 1.0, step: 0.1, default: 1.0},
    "colsample_bytree": {type: "float", min: 0.5, max: 1.0, step: 0.1, default: 1.0},
    "gamma": {type: "float", min: 0.0, max: 10.0, step: 0.1, default: 0.0},
    "eval_metric": {type: "select", options: ["logloss","mlogloss","auc"], default: "logloss"},
    "test_split": {type: "float", min: 0.1, max: 0.5, step: 0.1, default: 0.2},
    "random_state": {type: "int", default: 42}
  },

    // === Кластеризація ===
  "agglomerative": {
    "n_clusters": {type: "int", min: 2, max: 20, default: 2},
    "linkage": {type: "select", options: ["ward","complete","average","single"], default: "ward"},
    "metric": {type: "select", options: ["euclidean","manhattan","cosine"], default: "euclidean"}
  },
  "dbscan": {
    "eps": {type: "float", min: 0.1, max: 5.0, step: 0.1, default: 0.5},
    "min_samples": {type: "int", min: 2, max: 50, default: 5},
    "metric": {type: "select", options: ["euclidean","manhattan","cosine"], default: "euclidean"}
  },
  "gmm": {
    "n_components": {type: "int", min: 1, max: 20, default: 3},
    "covariance_type": {type: "select", options: ["full","tied","diag","spherical"], default: "full"},
    "max_iter": {type: "int", min: 50, max: 1000, default: 100},
    "n_init": {type: "int", min: 1, max: 10, default: 1},
    "random_state": {type: "int", default: 42}
  },
  "kmeans": {
    "n_clusters": {type: "int", min: 2, max: 50, default: 8},
    "init": {type: "select", options: ["k-means++","random"], default: "k-means++"},
    "max_iter": {type: "int", min: 100, max: 1000, default: 300},
    "n_init": {type: "int", min: 1, max: 50, default: 10},
    "random_state": {type: "int", default: 42}
  },
  "mean_shift": {
    "bandwidth": {type: "float", min: 0.1, max: 10.0, step: 0.1, default: null},
    "max_iter": {type: "int", min: 100, max: 1000, default: 300},
    "cluster_all": {type: "bool", default: true}
  },
  "optics": {
    "min_samples": {type: "int", min: 1, max: 50, default: 5},
    "max_eps": {type: "float", min: 0.1, max: 10.0, step: 0.1, default: 2.0},
    "metric": {type: "select", options: ["euclidean","manhattan","cosine"], default: "euclidean"}
  },
  "spectral": {
    "n_clusters": {type: "int", min: 2, max: 20, default: 8},
    "affinity": {type: "select", options: ["rbf","nearest_neighbors"], default: "rbf"},
    "assign_labels": {type: "select", options: ["kmeans","discretize"], default: "kmeans"},
    "n_neighbors": {type: "int", min: 2, max: 50, default: 10},
    "random_state": {type: "int", default: 42}
  }
};

let selectedColumn = null;
let selectedTargetColumn = null;
let taskType = null;
let currentModelName = null;
let task_type_clustering_no_target = false;


/**
 * Функція selectColumn
 * Етапи:
 * 1. Якщо вже була обрана колонка → зняти виділення.
 * 2. Виділити нову колонку.
 * 3. Зберегти назву вибраної колонки як target.
 * 4. Оновити UI з інформацією про вибір.
 * 5. Показати кнопку визначення задачі.
 * 6. Записати лог про вибір.
 */

function selectColumn(element, dtype) {
    if (selectedColumn) {
        selectedColumn.classList.remove("selected"); // зняти виділення з попередньої
    }

    element.classList.add("selected"); // виділити нову колонку
    selectedColumn = element;
    selectedTargetColumn = element.innerText;

    document.getElementById("task-info").innerHTML =
        "<strong>Обрана колонка:</strong> " + element.innerText;

    document.getElementById("determine-task-btn").style.display = "inline-block";
    logger("info", `Колонка '${element.innerText}' обрана як цільова`);
}

/**
 * Функція determineTask
 * Етапи:
 * 1. Перевірити, чи вибрана колонка.
 * 2. Відправити запит на бекенд для визначення типу задачі.
 * 3. Отримати відповідь з типом задачі.
 * 4. Динамічно створити кнопки для вибору задачі (регресія, класифікація, кластеризація).
 * 5. Завжди додати кнопку "кластеризація без змінної".
 * 6. Показати блок вибору задачі у UI.
 * 7. Записати лог про визначений тип задачі.
 */
async function determineTask() {
  if (!selectedColumn) {
    alert("Спочатку оберіть колонку!");
    logger("warn", "Спроба визначення задачі без вибраної колонки");
    return;
  }

  const response = await fetch("/professional_mode/detect_task", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({target: selectedTargetColumn})
  });
  const result = await response.json();

  const taskType = result.task_type;
  const taskButtonsDiv = document.getElementById("task-buttons");
  taskButtonsDiv.innerHTML = "";

  if (taskType === "regression") {
    taskButtonsDiv.innerHTML = `
      <button class="btn-orange" onclick="selectTask('regression')">Регресія</button>
      <button class="btn-blue" onclick="selectTask('clustering')">Кластеризація</button>
    `;
  } else if (taskType === "classification") {
    taskButtonsDiv.innerHTML = `
      <button class="btn-orange" onclick="selectTask('classification')">Класифікація</button>
      <button class="btn-blue" onclick="selectTask('clustering')">Кластеризація</button>
    `;
  } else if (taskType === "clustering") {
    taskButtonsDiv.innerHTML = `
      <button class="btn-blue" onclick="selectTask('clustering')">Кластеризація</button>
    `;
  }

  // Завжди додаємо кнопку кластеризації без змінної
  taskButtonsDiv.innerHTML += `
    <button class="btn-blue" onclick="selectClusteringNoTarget()">Кластеризація без змінної</button>
  `;

  document.getElementById("task-selection").style.display = "block";

  logger("info", `Тип задачі визначено: ${taskType}`);
}

/**
 * Функція selectClusteringNoTarget
 * Етапи:
 * 1. Викликати selectTask для кластеризації.
 * 2. Позначити, що кластеризація без target.
 * 3. Записати лог.
 */
function selectClusteringNoTarget() {
  selectTask('clustering');
  task_type_clustering_no_target = true;
  logger("info", "Обрана кластеризація без цільової змінної");
}

/**
 * Функція selectTask
 * Етапи:
 * 1. Зберегти вибраний тип задачі.
 * 2. Показати блок вибору моделі.
 * 3. Динамічно створити кнопки для доступних моделей залежно від задачі.
 * 4. Записати лог про вибір задачі.
 */
function selectTask(task) {
  taskType = task;
  document.getElementById("model-selection").style.display = "block";

  const modelButtonsDiv = document.getElementById("model-buttons");
  modelButtonsDiv.innerHTML = "";

  if (task === "regression") {
    modelButtonsDiv.innerHTML = `
      <button class="btn-purple" onclick="showParams('linear_regression')">Linear Regression</button>
      <button class="btn-purple" onclick="showParams('random_forest')">Random Forest</button>
      <button class="btn-purple" onclick="showParams('gradient_boosting')">Gradient Boosting</button>
      <button class="btn-purple" onclick="showParams('xgboost')">XGBoost</button>
    `;
  } else if (task === "classification") {
    modelButtonsDiv.innerHTML = `
      <button class="btn-purple" onclick="showParams('logistic_regression')">Logistic Regression</button>
      <button class="btn-purple" onclick="showParams('decision_tree')">Decision Tree</button>
      <button class="btn-purple" onclick="showParams('random_forest')">Random Forest</button>
      <button class="btn-purple" onclick="showParams('gradient_boosting')">Gradient Boosting</button>
      <button class="btn-purple" onclick="showParams('xgboost')">XGBoost</button>
      <button class="btn-purple" onclick="showParams('svm')">SVM</button>
      <button class="btn-purple" onclick="showParams('knn')">KNN</button>
    `;
  } else if (task === "clustering") {
    modelButtonsDiv.innerHTML = `
      <button class="btn-purple" onclick="showParams('kmeans')">KMeans</button>
      <button class="btn-purple" onclick="showParams('agglomerative')">Agglomerative</button>
      <button class="btn-purple" onclick="showParams('dbscan')">DBSCAN</button>
      <button class="btn-purple" onclick="showParams('optics')">OPTICS</button>
      <button class="btn-purple" onclick="showParams('gmm')">Gaussian Mixture (GMM)</button>
      <button class="btn-purple" onclick="showParams('mean_shift')">Mean Shift</button>
      <button class="btn-purple" onclick="showParams('spectral')">Spectral Clustering</button>
    `;
  }

  logger("info", `Обрана задача: ${task}`);
}

// Універсальна функція для показу параметрів моделі
function showParams(modelName) {
  // Зберігаємо назву вибраної моделі
  currentModelName = modelName;

  // Отримуємо форму для параметрів
  const form = document.getElementById("params-form");
  form.innerHTML = ""; // очищаємо попередні поля
  document.getElementById("params-block").style.display = "block"; // показуємо блок параметрів

  // Беремо конфіг параметрів для моделі з глобального словника
  const params = PARAMS_CONFIG[modelName];
  if (!params) {
    // Якщо параметрів немає → показуємо повідомлення
    form.innerHTML = "<p>❌ Немає параметрів для цієї моделі</p>";
    logger("warn", `Для моделі '${modelName}' немає конфігурації параметрів`);
    return;
  }

  // Генеруємо поля вводу для кожного параметра
  for (const [param, config] of Object.entries(params)) {
    let field = `<label>${param}:</label>`;

    // Якщо параметр числовий (int/float) → створюємо input type="number"
    if (config.type === "int" || config.type === "float") {
      field += `<input type="number" name="${param}" value="${config.default ?? ""}"
                 ${config.min ? `min="${config.min}"` : ""}
                 ${config.max ? `max="${config.max}"` : ""}
                 ${config.step ? `step="${config.step}"` : ""}>`;
    }
    // Якщо параметр булевий → створюємо select з True/False
    else if (config.type === "bool") {
      field += `<select name="${param}">
                  <option value="true" ${config.default ? "selected" : ""}>True</option>
                  <option value="false" ${!config.default ? "selected" : ""}>False</option>
                </select>`;
    }
    // Якщо параметр має список значень → створюємо select з options
    else if (config.type === "select") {
      field += `<select name="${param}">`;
      config.options.forEach(opt => {
        field += `<option value="${opt}" ${opt === config.default ? "selected" : ""}>${opt}</option>`;
      });
      field += `</select>`;
    }

    // Додаємо поле у форму
    form.innerHTML += `<div class="input-container">${field}</div>`;
  }

  logger("info", `Параметри для моделі '${modelName}' відображені у формі`);
}

// Збір параметрів з форми (сирі значення)
function collectFormParams(formId) {
  // Отримуємо форму за ID
  const form = document.getElementById(formId);

  // Створюємо FormData для збору значень
  const formData = new FormData(form);
  const params = {};

  // Проходимо по всіх полях форми
  for (const [key, value] of formData.entries()) {
    params[key] = value; // зберігаємо значення як є (Python сам нормалізує)
    logger("debug", `Зібрано параметр: ${key} = ${value}`);
  }

  logger("info", `Зібрано ${Object.keys(params).length} параметрів з форми '${formId}'`);
  return params;
}


// Запуск тренування
async function trainModel() {
  // Збираємо параметри з форми
  const form = document.getElementById("params-form");
  const formData = new FormData(form);
  const params = {};
  for (const [key, value] of formData.entries()) {
    params[key] = value;
  }

  // Формуємо payload
  const payload = {
    model_name: currentModelName,
    params: params,
    task_type: taskType
  };

  // Якщо кластеризація без цільової змінної — не додаємо target
  if (!task_type_clustering_no_target) {
    payload.target = selectedTargetColumn;
  } else {
    payload.target = null; // або взагалі не включати це поле
  }

  // Показуємо індикатор завантаження
  document.getElementById("loading").style.display = "block";

  try {
    const response = await fetch("/professional_mode/train", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });

    const result = await response.json();
    document.getElementById("loading").style.display = "none";

    let output = "<h3>Повна інформація про модель</h3>";

    // Основні дані
    output += `<p><strong>Повідомлення:</strong> ${result.message}</p>`;
    output += `<p><strong>Назва моделі:</strong> ${result.model_name}</p>`;
    output += `<p><strong>Тип задачі:</strong> ${result.task_type}</p>`;
    output += `<p><strong>Цільова колонка:</strong> ${result.target_column}</p>`;

    // Параметри
    if (result.params) {
      output += "<h4>Параметри моделі:</h4><ul>";
      for (const [param, val] of Object.entries(result.params)) {
        output += `<li>${param}: ${val}</li>`;
      }
      output += "</ul>";
    }

    // Ознаки
    if (result.features) {
      output += "<h4>Ознаки:</h4><ul>";
      result.features.forEach(f => {
        output += `<li>${f}</li>`;
      });
      output += "</ul>";
    }

    // Метрики
    if (result.metrics) {
      output += "<h4>Метрики:</h4><ul>";
      for (const [metric, val] of Object.entries(result.metrics)) {
        output += `<li>${metric}: ${val}</li>`;
      }
      output += "</ul>";
    }

    // Шаблон вводу
    if (result.input_template) {
      output += "<h4>Шаблон вводу:</h4><ul>";
      for (const [feature, placeholder] of Object.entries(result.input_template)) {
        output += `<li>${feature}: ${placeholder}</li>`;
      }
      output += "</ul>";
    }

    // Графіки
    if (result.plots && Object.keys(result.plots).length > 0) {
      output += "<h4>Графіки:</h4><div class='chart-container'>";
      for (const [plotName, base64img] of Object.entries(result.plots)) {
        output += `
          <div class="chart-block">
            <p>${plotName}</p>
            <img src="data:image/png;base64,${base64img}" class="chart" onclick="openOverlay(this.src)"/>
          </div>
        `;
      }
      output += "</div>";
    }
    if (result.model_name) {
      const filename = `${result.model_name}.pkl`;
      output += `<div class="result-section">
                   <h4><span class="icon">💾</span>Завантаження моделі</h4>
                   <button class="download-button" onclick="downloadModel('${filename}')">
                     ⬇️ Завантажити модель
                   </button>
                 </div>`;
    }

    document.getElementById("task-info").innerHTML = output;

    // Побудова форми для введення нових даних
    const inputFieldsDiv = document.getElementById("input-fields");
    inputFieldsDiv.innerHTML = "";

    if (result.task_type === "clustering" && result.input_template) {
      // Кластеризація → використовуємо input_template
      for (const [feature, placeholder] of Object.entries(result.input_template)) {
        const container = document.createElement("div");
        container.classList.add("input-container");
        container.innerHTML = `<label>${feature}</label><input name="${feature}" placeholder="${placeholder}" required>`;
        inputFieldsDiv.appendChild(container);
      }
      document.getElementById("input-form").style.display = "block";

    } else if (result.features && result.input_template) {
      // Регресія / класифікація → використовуємо features
      result.features.forEach(f => {
        const container = document.createElement("div");
        container.classList.add("input-container");
        container.innerHTML = `<label>${f}</label><input name="${f}" required>`;
        inputFieldsDiv.appendChild(container);
      });
      document.getElementById("input-form").style.display = "block";
    }


  } catch (error) {
    document.getElementById("loading").style.display = "none";
    alert("❌ Помилка при навчанні моделі: " + error);
  }
}

async function downloadModel(filename) {
  try {
    const response = await fetch(`/professional_mode/download_model/${filename}`);
    if (!response.ok) {
      throw new Error("Не вдалося завантажити файл");
    }
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = filename; // ім’я файлу при збереженні
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    alert("❌ Помилка при завантаженні: " + error.message);
  }
}

// Прогнозування на основі введених даних
async function predict(event) {
    event.preventDefault(); // блокуємо стандартну відправку форми

    const formData = new FormData(document.getElementById("predict-form"));
    const values = {};
    for (const [key, value] of formData.entries()) {
        values[key] = value;
    }

    const payload = {
        model_name: currentModelName,
        task_type: taskType,
        input_data: values
    };

    try {
        const response = await fetch("/professional_mode/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        let output = "<h3>Інформація про прогноз</h3>";

        output += `<div class="result-section">
                     <h4><span class="icon">⚙️</span>Технічна інформація</h4>
                     <ul>
                       <li>Назва моделі: ${result.model_name}</li>
                       <li>Тип задачі: ${result.task_type}</li>
                       <li>Тип моделі: ${result.model_type || "невідомо"}</li>
                     </ul>
                   </div>`;

        output += `<div class="result-section">
                     <h4><span class="icon">📥</span>Вхідні дані</h4>
                     <ul>`;
        for (const [key, val] of Object.entries(payload.input_data)) {
          output += `<li>${key}: ${val}</li>`;
        }
        output += `</ul></div>`;

        // Ознаки моделі
        if (result.features) {
          output += `<div class="result-section">
                       <h4><span class="icon">🧩</span>Ознаки моделі</h4>
                       <ul>`;
          result.features.forEach(f => {
            output += `<li>${f}</li>`;
          });
          output += `</ul></div>`;
        }

        // Побудова форми для введення нових даних
        if (result.task_type === "clustering" && result.input_template) {
          const inputFieldsDiv = document.getElementById("input-fields");
          inputFieldsDiv.innerHTML = "";

          for (const [feature, placeholder] of Object.entries(result.input_template)) {
            const container = document.createElement("div");
            container.classList.add("input-container");
            container.innerHTML = `<label>${feature}</label><input name="${feature}" placeholder="${placeholder}" required>`;
            inputFieldsDiv.appendChild(container);
          }

          document.getElementById("input-form").style.display = "block";
        } else if (result.features && result.input_template) {
          const inputFieldsDiv = document.getElementById("input-fields");
          inputFieldsDiv.innerHTML = "";
          result.features.forEach(f => {
            const container = document.createElement("div");
            container.classList.add("input-container");
            container.innerHTML = `<label>${f}</label><input name="${f}" required>`;
            inputFieldsDiv.appendChild(container);
          });

          document.getElementById("input-form").style.display = "block";
        }

        // Відображення прогнозу
        let predictionText = "";
        if (result.task_type === "regression") {
          predictionText = `Результат прогнозування: ${result.prediction}`;
        } else if (result.task_type === "classification") {
          predictionText = `Віднесено до класу: ${result.prediction}`;
        } else if (result.task_type === "clustering") {
          predictionText = `Віднесено до кластеру: ${result.prediction}`;
        }

        output += `<div class="result-section">
                     <h4><span class="icon">🔮</span>Результат</h4>
                     <p>${predictionText}</p>
                   </div>`;

        // Далі йде пайплайн
        if (result.pipeline_steps) {
          output += `<div class="result-section">
                       <h4><span class="icon">🛠️</span>Пайплайн</h4>
                       <table class="pipeline-table">
                         <tr><th>Крок</th><th>Опис</th></tr>`;
          result.pipeline_steps.forEach((step, index) => {
            output += `<tr><td>${index + 1}</td><td>${step}</td></tr>`;
          });
          output += `</table></div>`;
        }



        if (result.library_version) {
          output += `<div class="result-section">
                       <h4><span class="icon">📚</span>Версії бібліотек</h4>
                       <ul>`;
          for (const [lib, ver] of Object.entries(result.library_version)) {
            output += `<li>${lib}: ${ver}</li>`;
          }
          output += `</ul></div>`;
        }

        if (result.execution_time) {
          output += `<div class="result-section">
                       <h4><span class="icon">⏱️</span>Час виконання</h4>
                       <p>${result.execution_time}</p>
                     </div>`;
        }

        // створюємо кнопку для формування звіту
        let reportBtn = document.getElementById("generate-report-btn");
        if (!reportBtn) {
          reportBtn = document.createElement("button");
          reportBtn.id = "generate-report-btn";
          reportBtn.classList.add("btn-green");
          reportBtn.innerText = "Сформувати звіт (PDF)";
          reportBtn.onclick = () => generateReport(result);
          document.getElementById("prediction-result").appendChild(reportBtn);
        } else {
          reportBtn.style.display = "block";
          reportBtn.onclick = () => generateReport(result);
        }

        document.getElementById("prediction-result").innerHTML = output;

    } catch (error) {
        alert("❌ Помилка при запиті: " + error);
    }
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
  const response = await fetch("/professional_mode/generate_report", {
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

window.selectColumn = selectColumn;
window.determineTask = determineTask;
window.selectClusteringNoTarget = selectClusteringNoTarget;
window.selectTask = selectTask;
window.showParams = showParams;
window.trainModel = trainModel;
window.downloadModel = downloadModel;
window.predict = predict;
window.openOverlay = openOverlay;
window.closeOverlay = closeOverlay;
window.generateReport = generateReport;
