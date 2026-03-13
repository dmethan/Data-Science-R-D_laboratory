import { logger } from "./logger.js";

function goToBasic() {
    logger("debug", "Виклик функції goToBasic");
    window.location.href = "/basic_mode";
    logger("info", "⬅️ Перехід виконано: /basic_mode");
}

function goToProfessional() {
    logger("debug", "Виклик функції goToProfessional");
    window.location.href = "/professional_mode";
}

// Робимо функції доступними для HTML
window.goToBasic = goToBasic;
window.goToProfessional = goToProfessional;