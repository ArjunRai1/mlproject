from src.datascienceproject.logger import logging
from src.datascienceproject.exception import CustomException
from src.datascienceproject.components.data_ingestion import DataIngestion 
import sys

if __name__ == "__main__":
    logging.info("Execution has started")

    try:
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)