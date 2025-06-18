import logging
from pathlib import Path
from pydantic import validate_call

console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("jwstnoobfriend")
logger.setLevel(logging.INFO)

@validate_call
def get_console_handler(logger_level: int | None = None) -> logging.StreamHandler:
    console_handler = logging.StreamHandler()
    if logger_level is not None:
        console_handler.setLevel(logger_level)
    console_handler.setFormatter(console_formatter)
    return console_handler

@validate_call
def get_file_handler(log_file: Path, logger_level: int | None = None) -> logging.FileHandler:
    file_handler = logging.FileHandler(log_file)
    if logger_level is not None:
        file_handler.setLevel(logger_level)
    file_handler.setFormatter(file_formatter)
    return file_handler
