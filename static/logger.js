// logger.js — універсальна функція логування для фронтенду
export async function logger(level, message) {
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${level.toUpperCase()}: ${message}`;

  // Вивід у консоль для зручності
  if (console[level]) {
    console[level](logMessage);
  } else {
    console.log(logMessage);
  }

  // Відправка на бекенд FastAPI
  try {
    await fetch("/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ level, message })
    });
  } catch (err) {
    console.error("❌ Не вдалося відправити лог на сервер:", err);
  }
}
