from houseprediction.components.data_ingestion import DataIngestion
from houseprediction.entity.config_entity import TrainingPipelineConfig, DataingestionConfig
from houseprediction.logging.logger import logging
from houseprediction.exceptation.exceptation import housepredException
import os


if __name__ == "__main__":
    trainingpipelineconfig = TrainingPipelineConfig()
    dataingestionconfig =  DataingestionConfig(trainingpipelineconfig)
    dataingestion = DataIngestion(dataingestionconfig)
    logging.info("Starting data ingestion")
    dataingestion_artifact = dataingestion.initiate_data_ingestion()
    logging.info("Data ingestion completed")
    print(dataingestion_artifact)


