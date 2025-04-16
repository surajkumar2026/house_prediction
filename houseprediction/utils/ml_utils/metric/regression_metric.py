from houseprediction.exceptation.exceptation import housepredException
from houseprediction.logging.logger import logging
import sys
from houseprediction.entity.artifact_entity import RegressionMetricArtifact

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error

)


def get_Regression_score(y_true,y_pred)->RegressionMetricArtifact:
    try:
        model_r2_score = r2_score(y_true, y_pred)
        model_mse = mean_squared_error(y_true, y_pred)
        model_rmse = np.sqrt(model_mse)
        model_mae = mean_absolute_error(y_true, y_pred)

        regression_metric = RegressionMetricArtifact(r2_score=model_r2_score, MSE=model_mse, RMSE=model_rmse, MAE=model_mae)
        return regression_metric
       
    except Exception as e:
        raise housepredException(e,sys) from e
