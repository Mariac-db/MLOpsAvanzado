import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import mlflow
import os
import pickle

mlflow.set_tracking_uri("sqlite:///backend.db")
mlflow.set_experiment("nyc-taxi-experiment")

class TrainingModel: 
    """This class trains a model and logs the results in mlflow"""
    def __init__(self):
        self.X_train, self.y_train = self.load_pickle(os.path.join("data_model_input", "train.pkl"))
        self.X_val, self.y_val = self.load_pickle(os.path.join("data_model_input", "valid.pkl"))
        self.X_test, self.y_test = self.load_pickle(os.path.join("data_model_input", "test.pkl"))
       
    def load_pickle(self, filename:str):
        """this method loads a pickle file
        Args:
            filename (str): path to the pickle file
        Returns:
            object: the object contained in the pickle file"""
        with open(filename, "rb") as f_in:
            return pickle.load(f_in)
        
    def train_rfr_model(self, tag: str):
        """ this method trains a random forest regressor model and logs the results in mlflow
        Args:
            tag (str): tag to identify the model
        Returns:
            None"""
    
        with mlflow.start_run(run_name="randomforestregressor"):
            # Establece etiquetas o tags si es necesario
            mlflow.set_tag("developer", "camila")
            mlflow.set_tag("model", tag)

            params = {
                'n_estimators': 100, 
                'max_depth': 5,     
                'min_samples_split': 2, 
                'min_samples_leaf': 1,  
                'max_features': 'sqrt',  
                'bootstrap': True,      
                'random_state': 42,      
                'n_jobs': -1,           
                'verbose': 0  
            }

            rf = RandomForestRegressor(**params)
            rf.fit(self.X_train, self.y_train)
            val_rmse = mean_squared_error(self.y_val, rf.predict(self.X_val), squared=False)
            mlflow.log_metric("val_rmse", val_rmse)
            test_rmse = mean_squared_error(self.y_test, rf.predict(self.X_test), squared=False)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.sklearn.log_model(rf, "random_forest_model")
            mlflow.log_params(params)
            
    def train_xgboost_model(self, tag: str):
        """this method trains a xgboost model and logs the results in mlflow
        Args:
            tag (str): tag to identify the model
        Returns:
            None
        """
        with mlflow.start_run(run_name="xgboost"):
            mlflow.set_tag("developer", "camila")
            mlflow.set_tag("model", tag)

            # Definir los hiperparámetros que deseas ajustar
            params = {
                'max_depth': 4,  # Ejemplo de valores a probar para la profundidad máxima
                'learning_rate': 0.001,  # Ejemplo de tasas de aprendizaje a probar
            }
            train = xgb.DMatrix(self.X_train, label=self.y_train)
            valid = xgb.DMatrix(self.X_val, label=self.y_val)
            test = xgb.DMatrix(self.X_test, label=self.y_test)
            booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=50,
            evals=[(valid, 'validation')],
            early_stopping_rounds=3
            )
            y_val_pred = booster.predict(valid)
            y_test_pred = booster.predict(test)
            val_rmse = mean_squared_error(self.y_val, y_val_pred, squared=False)
            test_rmse = mean_squared_error(self.y_test, y_test_pred, squared=False)
    
            mlflow.log_params(params)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.sklearn.log_model(booster, "xgboost_model")

    
if __name__ == '__main__':
    training_model = TrainingModel()
    training_model.train_rfr_model(tag = "rfr")
    training_model.train_xgboost_model(tag="xgboost")
# mlflow ui --backend-store-uri sqlite:///backend.db --port 8080

    