from houseprediction.exceptation.exceptation import housepredException
import os
import sys


class HouseModel:

    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
            
        except Exception as e:
            raise housepredException(e, sys) from e
        

    def predict(self,x):
        try:
            x_transformed= self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transformed)
            return y_hat
        except Exception as e:
            raise housepredException(e, sys)