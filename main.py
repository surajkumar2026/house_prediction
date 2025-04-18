from houseprediction.components.data_ingestion import DataIngestion

from houseprediction.entity.config_entity import (TrainingPipelineConfig,
                                                   DataingestionConfig,
                                                   DataTransformationConfig,
                                                   ModelTrainerConfig,)
from houseprediction.logging.logger import logging
from houseprediction.exceptation.exceptation import housepredException
import os

from houseprediction.components.data_transformation import DataTransformation

from houseprediction.entity.artifact_entity import (DataIngestionArtifact,
                                                     DataTransformationArtifact,
                                                     ModelTrainerArtifact)
import sys

from houseprediction.components.model_trainer import ModelTrainer   


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

    datamodeltrainerconfig = ModelTrainerConfig(trainingpipelineconfig)
    datamodeltrainer       = ModelTrainer(datamodeltrainerconfig, data_transformation_artifact)
    model_trainer_artifact = datamodeltrainer.initiate_model_trainer()
    logging.info("Model training completed")
    print(model_trainer_artifact)

