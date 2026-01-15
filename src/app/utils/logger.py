import sys
from loguru import logger
import os

# Create logs directory if not exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure Loguru
# Remove default handler
logger.remove()

# Add Console Handler (Colorized, Info Level)
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add File Handler (Rotation 10MB, Retention 7 days)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    encoding="utf-8"
)

