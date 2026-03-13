import { logger } from "./logger.js";

function openOverlay(imgSrc) {
  const overlay = document.getElementById("image-overlay");
  const overlayImg = document.getElementById("overlay-img");
  overlay.style.display = "block";
  overlayImg.src = imgSrc;
}

function closeOverlay() {
  document.getElementById("image-overlay").style.display = "none";
}

document.querySelectorAll(".upload-form").forEach((form) => {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(form);

    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: formData
      });
      if (!response.ok) throw new Error("Не вдалося завантажити файл");

      const result = await response.json();
      const metadata = result.metadata;

      // створюємо контейнер для моделі
      const container = document.createElement("div");
      container.classList.add("model-card");

      // кнопка закриття
      const closeBtn = document.createElement("span");
      closeBtn.classList.add("close-btn");
      closeBtn.innerHTML = "&times;";
      closeBtn.onclick = () => container.remove();
      container.appendChild(closeBtn);

      let output = `<h3 class="model-title">${result.filename}</h3>`;
      output += `<div class="model-block task-block"><p><strong>Тип задачі:</strong> ${metadata.task_type}</p></div>`;
      output += `<div class="model-block target-block"><p><strong>Цільова колонка:</strong> ${metadata.target_column}</p></div>`;

      if (metadata.params) {
        output += `<div class="model-block params-block"><h4>Параметри моделі:</h4><ul>`;
        for (const [param, val] of Object.entries(metadata.params)) {
          output += `<li>${param}: ${val}</li>`;
        }
        output += `</ul></div>`;
      }

      if (metadata.features) {
        output += `<div class="model-block features-block"><h4>Ознаки:</h4><ul>`;
        metadata.features.forEach(f => {
          output += `<li>${f}</li>`;
        });
        output += `</ul></div>`;
      }

      if (metadata.metrics) {
        output += `<div class="model-block metrics-block"><h4>Метрики:</h4><ul>`;
        for (const [metric, val] of Object.entries(metadata.metrics)) {
          output += `<li>${metric}: ${val}</li>`;
        }
        output += `</ul></div>`;
      }

      if (metadata.input_template) {
        output += `<div class="model-block input-block"><h4>Шаблон вводу:</h4><ul>`;
        for (const [feature, placeholder] of Object.entries(metadata.input_template)) {
          output += `<li>${feature}: ${placeholder}</li>`;
        }
        output += `</ul></div>`;
      }

      if (metadata.plots && Object.keys(metadata.plots).length > 0) {
        output += `<div class="model-block plots-block"><h4>Графіки:</h4>`;
        for (const [plotName, base64img] of Object.entries(metadata.plots)) {
          output += `
            <div class="chart-block">
              <p>${plotName}</p>
              <img src="data:image/png;base64,${base64img}" class="chart" onclick="openOverlay(this.src)"/>
            </div>
          `;
        }
        output += `</div>`;
      }

      // додаємо контент після кнопки
      const contentDiv = document.createElement("div");
      contentDiv.innerHTML = output;
      container.appendChild(contentDiv);

      document.getElementById("models-info").appendChild(container);
    } catch (error) {
      alert("❌ Помилка: " + error.message);
    }
  });
});

window.closeOverlay = closeOverlay;
window.openOverlay = openOverlay;