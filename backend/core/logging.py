"""
Structured Logging

Production-ready logging with JSON output support for easy parsing.
"""

import sys
import logging
import json
from datetime import datetime
from typing import Any
from functools import lru_cache


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
        
        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter with colors."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"{color}[{timestamp}] {record.levelname:8}{self.RESET} | {record.name} | {record.getMessage()}"
        
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


class AppLogger(logging.Logger):
    """Extended logger with structured data support."""
    
    def _log_with_data(
        self,
        level: int,
        msg: str,
        data: dict[str, Any] | None = None,
        *args,
        **kwargs
    ):
        if data:
            extra = kwargs.get("extra", {})
            extra["extra_data"] = data
            kwargs["extra"] = extra
        
        self.log(level, msg, *args, **kwargs)
    
    def debug_with_data(self, msg: str, data: dict[str, Any] | None = None, **kwargs):
        self._log_with_data(logging.DEBUG, msg, data, **kwargs)
    
    def info_with_data(self, msg: str, data: dict[str, Any] | None = None, **kwargs):
        self._log_with_data(logging.INFO, msg, data, **kwargs)
    
    def warning_with_data(self, msg: str, data: dict[str, Any] | None = None, **kwargs):
        self._log_with_data(logging.WARNING, msg, data, **kwargs)
    
    def error_with_data(self, msg: str, data: dict[str, Any] | None = None, **kwargs):
        self._log_with_data(logging.ERROR, msg, data, **kwargs)


def setup_logging(
    level: str = "INFO",
    format_type: str = "text",
    name: str = "llm_ta"
) -> AppLogger:
    """
    Setup application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: "json" for structured logs, "text" for human-readable
        name: Logger name
    
    Returns:
        Configured logger instance
    """
    # Set custom logger class
    logging.setLoggerClass(AppLogger)
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Set formatter
    if format_type == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())
    
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


@lru_cache()
def get_logger(name: str = "llm_ta") -> AppLogger:
    """Get a cached logger instance."""
    from core.config import get_settings
    
    settings = get_settings()
    return setup_logging(
        level=settings.log_level,
        format_type=settings.log_format,
        name=name
    )
