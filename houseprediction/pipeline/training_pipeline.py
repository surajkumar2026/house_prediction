import os
import sys

from houseprediction.exceptation.exceptation import housepredException
from houseprediction.logging.logger import logging


from houseprediction.components.data_ingestion import DataIngestion
       
from houseprediction.components.data_transformation import DataTransformation
from houseprediction.components.model_trainer import ModelTrainer

from houseprediction.entity.config_entity import (
    TrainingPipelineConfig,
    DataingestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from houseprediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact ,
)

from houseprediction.constant.training_pipeline import TRAINING_BUCKET_NAME
from houseprediction.constant.training_pipeline import SAVED_MODEL_DIR

class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
            
        except Exception as e:
            raise housepredException(e, sys)
        

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataingestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info(f"start Data ingestion ")
            
            data_ingestion = DataIngestion(dataingestion_config= self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed")

            return data_ingestion_artifact
          
        except Exception as e:
            raise housepredException(e, sys)
        

    def start_data_transformation(self, data_ingestion_artifact:DataIngestionArtifact)->DataTransformationArtifact:
        try:
            self.data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info(f"start Data transformation")

            data_transformation = DataTransformation(dataingestion_artifact= data_ingestion_artifact,
                                                          data_transformation_config=self.data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed")
            return data_transformation_artifact     
           
        except Exception as e:
            raise housepredException(e, sys)
        


    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            self.model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info(f"start Model trainer")
            model_trainer = ModelTrainer(model_trainer_config= self.model_trainer_config,
                                          data_tarnsformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model trainer completed")
            return model_trainer_artifact
            
        except Exception as e:
            raise housepredException(e, sys)
        


    def run_pipeline(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Training pipeline started")
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)  
            model_tariner_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            logging.info(f"Training pipeline completed")

            return model_tariner_artifact


        except Exception as e:
            raise housepredException(e, sys)