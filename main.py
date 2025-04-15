from houseprediction.components.data_ingestion import DataIngestion

from houseprediction.entity.config_entity import TrainingPipelineConfig, DataingestionConfig, DataTransformationConfig
from houseprediction.logging.logger import logging
from houseprediction.exceptation.exceptation import housepredException
import os

from houseprediction.components.data_transformation import DataTransformation

from houseprediction.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
import sys


if __name__ == "__main__":
    trainingpipelineconfig = TrainingPipelineConfig()
    dataingestionconfig =  DataingestionConfig(trainingpipelineconfig)
    dataingestion = DataIngestion(dataingestionconfig)
    logging.info("Starting data ingestion")
    dataingestion_artifact = dataingestion.initiate_data_ingestion()
    logging.info("Data ingestion completed")
    print(dataingestion_artifact)

    datatransformationconfig = DataTransformationConfig(trainingpipelineconfig)
    datatransformation = DataTransformation(dataingestion_artifact, datatransformationconfig)
    data_transformation_artifact = datatransformation.initiate_data_transformation()
    logging.info("Data transformation completed")
    print(data_transformation_artifact)



