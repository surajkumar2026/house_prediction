import pickle
import numpy as np

from houseprediction.exceptation.exceptation import housepredException
from houseprediction.logging.logger import logging
import os
import sys  

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise housepredException(e, sys) from e
    

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise housepredException(e, sys) from e
    
def evaluate_models(X_train, Y_train, X_test, Y_test, models, params):
   
    try:
        report = {}
        for i in range (len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv =3)  
            gs.fit(X_train, Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, Y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score =r2_score(Y_train, y_train_pred)
            test_model_score =r2_score(Y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report



    except Exception as e:
        raise housepredException(e, sys) from e
    

def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise housepredException(e, sys) from e
    

    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise housepredException(e, sys) from e