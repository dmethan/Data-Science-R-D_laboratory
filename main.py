import time
from logging_config import logger, js_logger
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routers.basic_router import router as basic_router
from routers.router_professional import router as professional_router


app = FastAPI(title="ML Web Service")

# Підключення static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Підключення templates
templates = Jinja2Templates(directory="templates")

# Підключення роутерів
app.include_router(basic_router)
app.include_router(professional_router)

# ------------------------------
# Middleware для логування
# ------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"➡️ Запит: {request.method} {request.url}")
    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    logger.info(f"⬅️ Відповідь: статус={response.status_code}, час={process_time:.2f}мс")

    return response

# ------------------------------
# Головна сторінка
# ------------------------------
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    logger.info("=== Запуск home_page ===")
    response = templates.TemplateResponse(
        "main_index.html",
        {"request": request}
    )
    logger.info("=== Кінець home_page ===")
    return response

@app.post("/log")
async def log_message(payload: dict = Body(...)):
    level = payload.get("level", "info").lower()
    message = payload.get("message", "")

    if level == "debug":
        js_logger.debug(message)
    elif level == "error":
        js_logger.error(message)
    elif level == "warn":
        js_logger.warning(message)
    else:
        js_logger.info(message)

    return {"status": "ok"}
