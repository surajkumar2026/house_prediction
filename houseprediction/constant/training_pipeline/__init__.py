import os 
import sys


"""defining conatant variable for the training pipeline"""
TARGET_COLUMN = "Result"
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


 