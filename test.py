
from utils.wraps import save_logger

from loguru import logger


@save_logger
def main():
    logger.info("Hello, World!")
    logger.debug("debug message")
    logger.info("info message")
    logger.error("error message")
    
if __name__ == "__main__":
    main()