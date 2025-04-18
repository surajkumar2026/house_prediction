
from houseprediction.entity.config_entity import ModelTrainerConfig
from houseprediction.entity.artifact_entity import ModelTrainerArtifact, RegressionMetricArtifact, DataTransformationArtifact

from houseprediction.exceptation.exceptation import housepredException
from houseprediction.logging.logger import logging
import sys
import os


from houseprediction.utils.main_util.utils import evaluate_models,save_object,load_object, save_numpy_array_data, load_numpy_array_data
from houseprediction.utils.ml_utils.metric.regression_metric import get_Regression_score

from houseprediction.utils.ml_utils.model.estimator import HouseModel


from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from sklearn.ensemble import(
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
    ExtraTreesRegressor
)

import mlflow

class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_tarnsformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_tarnsformation_artifact
        except Exception as e:
            raise housepredException(e, sys)




    def track_mlflow(self, best_model, regressionmetrics):
        try:
            with mlflow.start_run():
                r2_score = regressionmetrics.r2_score
                mse = regressionmetrics.MSE
                rmse = regressionmetrics.RMSE
                mae = regressionmetrics.MAE

                mlflow.log_metric("r2_score", r2_score)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)

                mlflow.sklearn.log_model(best_model, "model")
          
        except Exception as e:
            raise housepredException(e, sys)
        

    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "ExtraTreesRegressor": ExtraTreesRegressor()
            }
            


            params= {

                "LinearRegression": {
                    #'copy_X', 'fit_intercept', 'n_jobs', 'positive'
                    'fit_intercept': [True],
                    'copy_X': [True],
                    'n_jobs': [None],
                    'positive': [False]
                    
                },
                
                "DecisionTreeRegressor": {
                    "max_depth": [10],
                    "min_samples_split": [5],
                    "min_samples_leaf": [2]
                },
                "RandomForestRegressor": {
                    "n_estimators": [100],
                    "max_depth": [10],
                    "min_samples_split": [5],
                    "min_samples_leaf": [2]
                },
                "GradientBoostingRegressor": {
                    "n_estimators": [100],
                    "learning_rate": [0.],
                    "max_depth": [3],
                    "min_samples_split": [2],
                    "min_samples_leaf": [1]
                },
                "XGBRegressor": {
                    "n_estimators": [100],
                    "learning_rate": [0.1],
                    "max_depth": [3],
                    "min_child_weight": [1]
                },
                "ExtraTreesRegressor": {
                    "n_estimators": [100],
                    "max_depth": [None],
                    "min_samples_split": [2],
                    "min_samples_leaf": [1]
                }

            }

            model_report :dict = evaluate_models(x_train, y_train, x_test, y_test, models, params)

            #best model score find krenge
            best_model_score = max(sorted(model_report.values()))

            #best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            logging.info(f"best model name is {best_model_name} and best model score is {best_model_score}")
            #best model

            best_model = models[best_model_name]

            y_train_pred = best_model.predict(x_train)

            regression_train_metrics = get_Regression_score(y_true=y_train, y_pred=y_train_pred)
            # track mlflow
            self.track_mlflow(best_model, regression_train_metrics)
            logging.info(f"train metrics are {regression_train_metrics}")

            y_test_pred = best_model.predict(x_test)
            regression_test_metrics = get_Regression_score(y_true=y_test, y_pred=y_test_pred)
            # track mlflow
            self.track_mlflow(best_model, regression_test_metrics)
            logging.info(f"test metrics are {regression_test_metrics}")

            # for new testing data, 
            preprocessor = load_object(file_path= self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_dir)
            os.makedirs(model_dir_path, exist_ok=True)

            House_model = HouseModel(preprocessor=preprocessor, model= best_model)
             
            save_object(self.model_trainer_config.trained_model_dir, obj=House_model) 
            
            # model pusher
            save_object("final_model/model.pkl",best_model)

             #model trainer Artifact
            model_trainer_artiifact= ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_dir,
                                                          train_metric_artifact=regression_train_metrics,
                                                          test_metric_artifact=regression_test_metrics)

            logging.info(f"model trainer artifcat:{model_trainer_artiifact}")
            return model_trainer_artiifact

        except Exception as e:
            raise housepredException(e, sys)
        

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array 
            train_arr = load_numpy_array_data(train_file_path)
            test_arr   = load_numpy_array_data(test_file_path)

            x_train , y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )

            mode_trainer_artifact= self.train_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            return mode_trainer_artifact


            
        except Exception as e:
            raise housepredException(e,sys)