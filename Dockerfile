# Базовий образ Python
FROM python:3.13-slim

# Встановлюємо робочу директорію
WORKDIR /app

# Копіюємо файли проекту
COPY . .

# Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# Команда запуску
# Запускаємо FastAPI через uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]