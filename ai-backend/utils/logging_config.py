"""Logging configuration for SportsScribe pipeline.

This module provides centralized logging configuration for all pipeline components.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO", log_file: str | None = None, include_debug: bool = False
) -> None:
    """Setup logging configuration for the SportsScribe pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        include_debug: Whether to include debug logs in file output
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if include_debug else numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG if include_debug else numeric_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels
    loggers_to_configure = [
        "scriber_agents.pipeline",
        "scriber_agents.data_collector",
        "scriber_agents.researcher",
        "scriber_agents.writer",
        "openai",
        "aiohttp",
        "urllib3",
    ]

    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG if include_debug else numeric_level)
        logger.propagate = True

    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    logging.info(
        f"🔧 Logging configured - Level: {level}, File: {log_file or 'None'}, Debug: {include_debug}"
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_pipeline_start(operation: str, **kwargs) -> None:
    """Log the start of a pipeline operation.

    Args:
        operation: Name of the operation
        **kwargs: Additional context information
    """
    logger = logging.getLogger("scriber_agents.pipeline")
    context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"[PIPELINE] Starting {operation} - {context}")


def log_pipeline_step(step: str, **kwargs) -> None:
    """Log a pipeline step.

    Args:
        step: Name of the step
        **kwargs: Additional context information
    """
    logger = logging.getLogger("scriber_agents.pipeline")
    context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"[PIPELINE] Step: {step} - {context}")


def log_pipeline_success(operation: str, duration: float, **kwargs) -> None:
    """Log successful completion of a pipeline operation.

    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **kwargs: Additional context information
    """
    logger = logging.getLogger("scriber_agents.pipeline")
    context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(
        f"[PIPELINE] {operation} completed successfully in {duration:.2f}s - {context}"
    )


def log_pipeline_error(
    operation: str, error: Exception, duration: float, **kwargs
) -> None:
    """Log an error in a pipeline operation.

    Args:
        operation: Name of the operation
        error: The exception that occurred
        duration: Duration in seconds
        **kwargs: Additional context information
    """
    logger = logging.getLogger("scriber_agents.pipeline")
    context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.error(
        f"[PIPELINE] {operation} failed after {duration:.2f}s - {error} - {context}"
    )


def log_data_collection(source: str, **kwargs) -> None:
    """Log data collection operations.

    Args:
        source: Data source name
        **kwargs: Additional context information
    """
    logger = logging.getLogger("scriber_agents.data_collector")
    context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"[COLLECTOR] Collecting from {source} - {context}")


def log_research_operation(operation: str, **kwargs) -> None:
    """Log research operations.

    Args:
        operation: Research operation name
        **kwargs: Additional context information
    """
    logger = logging.getLogger("scriber_agents.researcher")
    context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"[RESEARCHER] {operation} - {context}")


def log_writing_operation(article_type: str, **kwargs) -> None:
    """Log writing operations.

    Args:
        article_type: Type of article being written
        **kwargs: Additional context information
    """
    logger = logging.getLogger("scriber_agents.writer")
    context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"[WRITER] Generating {article_type} - {context}")


# Default configuration
if __name__ == "__main__":
    setup_logging(level="INFO", include_debug=False)
