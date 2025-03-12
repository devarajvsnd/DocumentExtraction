from loguru import logger
import os
from config import LOGS_DIR

# Configure logger
logger.add(
    os.path.join(LOGS_DIR, "app.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
    rotation="10 MB",
    retention="7 days",
)