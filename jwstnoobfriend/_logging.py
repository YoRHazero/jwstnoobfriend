import logging
console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Receive the level of logging from the user
def set_logging_level(logger, level = logging.INFO):
    """
    Set the logging level for the module.
    """
    logger.setLevel(level)
    logger.debug(f"Logging level set to {level}")
    return logger

# Receive the output log file from the user, output to console if not provided.
# output to console and file if provided.
def set_logging_file(logger, log_file = None):
    """
    Set the logging file for the module.
    """
    logger.handlers.clear()  # Clear existing handlers
    cn_handler = logging.StreamHandler()
    cn_handler.setFormatter(console_formatter)
    logger.addHandler(cn_handler)
    logger.debug("Logging to console")
    
    if log_file:
        fl_handler = logging.FileHandler(log_file)
        fl_handler.setFormatter(file_formatter)
        logger.addHandler(fl_handler)
        logger.debug(f"Logging to file {log_file}")
        
    return logger