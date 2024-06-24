import logging


def setup_logger():
    logger_ = logging.getLogger('chat-data')
    logger_.setLevel(logging.INFO)
    if not logger_.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s - [Thread ID: %(thread)d]'
        )
        console_handler.setFormatter(formatter)
        logger_.addHandler(console_handler)
    return logger_


logger = setup_logger()
