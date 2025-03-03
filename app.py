from src.datascienceproject.logger import logging
from src.datascienceproject.exception import CustomException
import sys

if __name__ == "__main__":
    logging.info("Execution has started")

    try:
        a=1/0
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)