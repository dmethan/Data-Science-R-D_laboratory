import logging
import os

# --- Папка logs ---
os.makedirs("logs", exist_ok=True)

# --- Кастомний FileHandler з обмеженням рядків ---
class LimitedFileHandler(logging.FileHandler):
    def __init__(self, filename, max_lines=500, **kwargs):
        super().__init__(filename, **kwargs)
        self.max_lines = max_lines
        # Очищаємо файл при старті
        open(filename, "w", encoding=kwargs.get("encoding", "utf-8")).close()

    def emit(self, record):
        super().emit(record)
        # Після запису, перевіряємо кількість рядків
        try:
            with open(self.baseFilename, "r", encoding=self.encoding) as f:
                lines = f.readlines()
            if len(lines) > self.max_lines:
                # Залишаємо останні max_lines рядків
                with open(self.baseFilename, "w", encoding=self.encoding) as f:
                    f.writelines(lines[-self.max_lines:])
        except Exception:
            pass  # На випадок проблем з файлом, не зупиняємо логер

# --- Формат повідомлень ---
log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
formatter = logging.Formatter(log_format)

# --- Хендлери для Python ---
info_handler = LimitedFileHandler("logs/app_info.log", mode="a", encoding="utf-8", max_lines=500)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

debug_handler = LimitedFileHandler("logs/app_debug.log", mode="a", encoding="utf-8", max_lines=500)
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# --- Хендлери для JS ---
js_info_handler = LimitedFileHandler("logs/js_info.log", mode="a", encoding="utf-8", max_lines=500)
js_info_handler.setLevel(logging.INFO)
js_info_handler.setFormatter(formatter)

class NoDebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.DEBUG

js_info_handler.addFilter(NoDebugFilter())

js_debug_handler = LimitedFileHandler("logs/js_debug.log", mode="a", encoding="utf-8", max_lines=500)
js_debug_handler.setLevel(logging.DEBUG)
js_debug_handler.setFormatter(formatter)

# --- Логери ---
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)
logger.addHandler(info_handler)
logger.addHandler(debug_handler)
logger.addHandler(console_handler)

js_logger = logging.getLogger("js")
js_logger.setLevel(logging.DEBUG)
js_logger.addHandler(js_info_handler)
js_logger.addHandler(js_debug_handler)
js_logger.addHandler(console_handler)
