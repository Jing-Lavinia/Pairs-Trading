"""
utils.py
--------
Global logging setup for the crypto statistical arbitrage pipeline.
Configures a single root logger that writes to both the console and a
persistent log file, ensuring all modules share a consistent log format.
"""
import logging
import os


def setup_global_logger(log_file_path: str = "logs/strategy_execution.log") -> logging.Logger:
    """
    Initialize global logging with synchronised console and file output.

    Args:
        log_file_path: Destination path for the persistent log file.

    Returns:
        Root logger instance configured for strategy-level tracing.
    """
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)
