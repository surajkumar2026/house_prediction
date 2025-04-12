
from houseprediction.entity.config_entity import DataingestionConfig
from houseprediction.entity.artifact_entity import DataIngestionArtifact
from houseprediction.exceptation.exceptation import housepredException
from houseprediction.logging.logger import logging

import numpy as np
from pymongo import MongoClient
import pandas as pd

from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
import sys
from pymongo import MongoClient

load_dotenv()

MONGO_DB_URL= os.getenv("MONGO_DB_URL")

class DataIngestion :
    def __init__(self,dataingestion_config:DataingestionConfig):
        try:
            self.dataingestion_config=dataingestion_config

        except Exception as e:
            raise housepredException(e,sys)
        

    def export_collection_as_dataframe(self):
        try:
            database_name= self.dataingestion_config.database_name
            collection_name= self.dataingestion_config.collection_name
            logging.info(f"Exporting collection {collection_name} from database {database_name} as dataframe")
            self.mongo_client= MongoClient(MONGO_DB_URL)
            collection= self.mongo_client[database_name][collection_name]
            logging.info(f"Collection {collection_name} from database {database_name} is connected")

            logging.info(f"Exporting collection {collection_name} from database {database_name} as dataframe")

            df = pd.DataFrame(list(collection.find()))

            if "exactPrice" in df.columns.to_list():
                df= df.drop(columns=["exactPrice"],axis=1)

            logging.info(f"Exported collection {collection_name} from database {database_name} as dataframe")

            return df 

        except Exception as e:
            raise housepredException(e,sys)
        



    def save_data_to_feature_store(self, dataframe:pd.DataFrame):
        try:
          
            feature_store_path = self.dataingestion_config.feature_store_dir
            logging.info(f"Saving data to feature store at {feature_store_path}")

            dir_path = os.path.dirname(feature_store_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Directory {dir_path} created")

            dataframe.to_csv(feature_store_path, index=False, header=True)  

            return dataframe
        
        except Exception as e:
            raise housepredException(e,sys)
        

    def split_data_as_train_test(self, dataframe:pd.DataFrame):
        try:
            logging.info(f"Splitting data into train and test")
            
            train_set, test_set = train_test_split(dataframe, test_size=self.dataingestion_config.train_test_split_ratio, random_state=42)

            logging.info(f"Train set shape: {train_set.shape}")
            logging.info(f"Test set shape: {test_set.shape}")

            logging.info(f"Train set and test set split successfully")

            dir_path = os.path.dirname(self.dataingestion_config.training_file_path)

            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"for test and train Directory {dir_path} created")

            train_set.to_csv(self.dataingestion_config.training_file_path, index=False, header=True)

            test_set.to_csv(self.dataingestion_config.testing_file_path, index=False, header=True)
            logging.info(f"Train and test set saved successfully")


            return train_set, test_set
        
        except Exception as e:
            raise housepredException(e,sys)
        

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:

            logging.info(f"Initiating data ingestion")
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.save_data_to_feature_store(dataframe)

            self.split_data_as_train_test(dataframe)

            dataingestionartifact=DataIngestionArtifact(train_file_path=self.dataingestion_config.training_file_path,
                                                        test_file_path=self.dataingestion_config.testing_file_path)
            logging.info(f"Data ingestion completed successfully")
            return dataingestionartifact
        
        except Exception as e:
            raise housepredException(e,sys)


