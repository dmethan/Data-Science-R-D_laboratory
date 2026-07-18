let currentType = null;
let currentPage = 1;
let linesPerPage = 100;
let allLines = [];

export async function loadLogs(type, page = 1) {
  currentType = type;
  currentPage = page;

  try {
    const response = await fetch(`/professional_mode/logs?type=${type}`);
    const data = await response.json();
    allLines = data.lines || [];

    // Налаштовуємо фільтр рівнів
    setupFilter(type);

    renderLogs();
    renderPagination();

    // Логування дії користувача
    await fetch("/log", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ level: "info", message: `Натиснута кнопка: ${type}` })
    });

  } catch (err) {
    console.error("❌ Помилка при завантаженні логів:", err);
  }
}

function renderLogs() {
  const container = document.getElementById("log-container");
  container.innerHTML = "";

  const searchTerm = document.getElementById("search-input").value.toLowerCase();
  const filterLevel = document.getElementById("filter-select").value;

  // Пагінація
  const start = (currentPage - 1) * linesPerPage;
  const end = start + linesPerPage;
  let pageLines = allLines.slice(start, end);

  pageLines.forEach(line => {
    if (filterLevel && !line.includes(filterLevel.toUpperCase())) return;

    const div = document.createElement("div");
    div.classList.add("log-line");

    if (line.includes("INFO")) div.classList.add("INFO");
    else if (line.includes("DEBUG")) div.classList.add("DEBUG");
    else if (line.includes("ERROR")) div.classList.add("ERROR");
    else if (line.includes("WARNING")) div.classList.add("WARNING");

    // Підсвічування пошуку
    if (searchTerm && line.toLowerCase().includes(searchTerm)) {
      const regex = new RegExp(`(${searchTerm})`, "gi");
      div.innerHTML = line.replace(regex, `<span class="highlight">$1</span>`);
    } else {
      div.textContent = line;
    }

    container.appendChild(div);
  });
}

function renderPagination() {
  const pagination = document.getElementById("pagination");
  pagination.innerHTML = "";

  const totalPages = Math.ceil(allLines.length / linesPerPage);
  if (totalPages <= 1) return;

  for (let i = 1; i <= totalPages; i++) {
    const btn = document.createElement("button");
    btn.textContent = i;
    btn.classList.add("page-btn");
    if (i === currentPage) btn.style.fontWeight = "bold";
    btn.onclick = () => {
      currentPage = i;
      renderLogs();
      renderPagination();
    };
    pagination.appendChild(btn);
  }
}

function setupFilter(type) {
  const filterSelect = document.getElementById("filter-select");
  filterSelect.innerHTML = "";

  const levels = type.includes("debug")
    ? ["", "info", "warn", "error", "debug"]
    : ["", "info", "warn", "error"];

  levels.forEach(level => {
    const option = document.createElement("option");
    option.value = level;
    option.textContent = level ? level.toUpperCase() : "Всі рівні";
    filterSelect.appendChild(option);
  });

  filterSelect.onchange = renderLogs;
}

// // Автооновлення кожні 5 секунд
// setInterval(() => {
//   if (currentType) loadLogs(currentType, currentPage);
// }, 5000);

// Пошук у реальному часі
document.getElementById("search-input").addEventListener("input", renderLogs);

window.loadLogs = loadLogs
window.renderLogs = renderLogs
window.renderPagination = renderPagination
window.setupFilter = setupFilter