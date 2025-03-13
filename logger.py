from loguru import logger
import os
from pathlib import Path
from config import LOGS_DIR

# Ensure the logs directory exists
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

# Remove any existing default logger configuration
logger.remove()

# Add a file logger
logger.add(
    os.path.join(LOGS_DIR, "app.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
    rotation="10 MB",  # Rotate logs after 10 MB
    retention="7 days",  # Keep logs for 7 days
    encoding="utf-8",
    enqueue=True,  # Ensure thread-safe logging
)

# Add a console logger (for Streamlit compatibility)
logger.add(
    sink=lambda msg: print(msg, end=""),  # Log to terminal
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)