import logging
def get_logger(name: str = "quantum_portfolio") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    return logger
