from functools import wraps

import datetime
from loguru import logger

def print_return(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result!r}")
        return result
    return wrapper


def logger_return(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} returned: {result!r}")
        return result
    return wrapper


def save_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        filename = f"../output/output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(filename, level="DEBUG", encoding="utf-8", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        result = func(*args, **kwargs)
        return result
       
    return wrapper