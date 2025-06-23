import logging
from pathlib import Path
from pydantic import validate_call
from typing import Literal

LoggerLevel = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

base_logger = logging.getLogger("jwstnoobfriend")
base_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@validate_call
def get_console_handler(logger_level: int | None = None, logger_level_name: LoggerLevel | None = None) -> logging.StreamHandler:
    console_handler = logging.StreamHandler()
    if logger_level is not None:
        console_handler.setLevel(logger_level)
    elif logger_level_name is not None:
        console_handler.setLevel(getattr(logging, logger_level_name))
    else:
        pass
    console_handler.setFormatter(console_formatter)
    return console_handler

@validate_call
def get_file_handler(log_file: Path, logger_level: int | None = None, logger_level_name: LoggerLevel | None = None) -> logging.FileHandler:
    file_handler = logging.FileHandler(log_file)
    if logger_level is not None:
        file_handler.setLevel(logger_level)
    elif logger_level_name is not None:
        file_handler.setLevel(getattr(logging, logger_level_name))
    else:
        pass
    file_handler.setFormatter(file_formatter)
    return file_handler

@validate_call
def getLogger(name: str) -> logging.Logger:
    """The same as `logging.getLogger(name)`"""
    return logging.getLogger(name)