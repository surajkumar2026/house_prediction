import os 
import sys
import numpy as np

"""defining conatant variable for the training pipeline"""
TARGET_COLUMN = "exactPrice"
PIPELINE_NAME: str = "HousePricePrediction"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "house_price.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR =  os.path.join("saved_models")
MODEL_FILE_NAME =  "model.pkl"


"""defining conatant variable for the data ingestion"""
DATA_INGESTION_DATABASE_NAME: str = "house_price_dataset"
DATA_INGESTION_COLLECTION_NAME:str = "HouseData"
DATA_INGESTION_DIR_NAME :str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2




"""defining conatant variable for the data transformation"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

## kkn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"

"""defining conatant variable for the model trainer"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05

TRAINING_BUCKET_NAME = "netwworksecurity"

 