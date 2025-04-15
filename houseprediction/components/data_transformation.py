from houseprediction.exceptation.exceptation import housepredException
from houseprediction.logging.logger import logging
from houseprediction.entity.config_entity import DataTransformationConfig, DataingestionConfig
from houseprediction.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact

from houseprediction.utils.main_util.utils import save_numpy_array_data, save_object

from houseprediction.constant.training_pipeline import TARGET_COLUMN


from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline
import sys
import pandas as pd
import numpy as np 

class DataTransformation:
    def __init__(self, dataingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.dataingestion_artifact = dataingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise housepredException(e, sys)
        

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise housepredException(e, sys)
        

    def get_data_transformer_object(cls) -> Pipeline:
        try:
           processor: Pipeline = Pipeline([("scaler", MinMaxScaler())])
           

           return processor
        except Exception as e:
            raise housepredException(e, sys)

        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation started")
            train_df = self.read_data(self.dataingestion_artifact.train_file_path)
            test_df = self.read_data(self.dataingestion_artifact.test_file_path)

            input_train_data = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_data= np.log(train_df[TARGET_COLUMN])

            input_test_data = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_data= np.log(test_df[TARGET_COLUMN])

            prepriocessor = self.get_data_transformer_object()

            preprocessor_object = prepriocessor.fit(input_train_data)

            transformed_train_input_data = preprocessor_object.transform(input_train_data)
            transformed_test_input_data = preprocessor_object.transform(input_test_data)

            train_arr= np.c_[transformed_train_input_data,np.array(target_feature_train_data)]
            test_arr= np.c_[transformed_test_input_data,np.array(target_feature_test_data)]

            # save the transformed data to csv files
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

            Datatransformationartifcat = DataTransformationArtifact(transformed_object_file_path= self.data_transformation_config.transformed_object_file_path,
                                                                    transformed_train_file_path= self.data_transformation_config.transformed_train_file_path,
                                                                    transformed_test_file_path= self.data_transformation_config.transformed_test_file_path)
            
            return DataTransformationArtifact




            
        except Exception as e:
            raise housepredException(e, sys)