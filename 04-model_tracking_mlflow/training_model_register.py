import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import os
import pickle

class TrainingModel: 
    """
    A class for training machine learning models and tracking experiments using MLflow.

    Attributes:
        RF_PARAMS (list): List of random forest hyperparameters to track.

    Methods:
        load_pickle(filename):
            Load a pickled object from a file.

        train_model_and_track(data_input="data_model_input"):
            Train a random forest model and track the experiment using MLflow.

        register_model():
            Register the trained model.
    """
    def __init__(self):
        self.RF_PARAMS = ['max_depth', 
            'n_estimators', 
            'min_samples_split', 
            'min_samples_leaf', 
            'random_state', 
            'n_jobs']
        #self.hpo_experiment_name = "mlflow_rfc"
       
    def load_pickle(self, filename:str):
        """
        Load a pickled object from a file.

        Args:
            filename (str): The name of the input pickle file.

        Returns:
            object: The loaded object.
        """
        with open(filename, "rb") as f_in:
            return pickle.load(f_in)
        
    def train_model_and_track(self, data_input = "data_model_input"):
        """
        Train a random forest model and track the experiment using MLflow.

        Args:
            data_input (str, optional): The path to preprocessed data. Defaults to "data_model_input".
        Returns:
            None
        """
        X_train, y_train = self.load_pickle(os.path.join(data_input, "train.pkl"))
        X_val, y_val = self.load_pickle(os.path.join(data_input, "valid.pkl"))
        X_test, y_test = self.load_pickle(os.path.join(data_input, "test.pkl"))

        with mlflow.start_run():
            #set developer
            mlflow.set_tag("developer", "camila")
            mlflow.log_param("data_row", "data/")
            mlflow.log_param("data_preprocessed", "data_model_input/")

            for param in self.RF_PARAMS:
                params[param] = int(params[param])

            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
            mlflow.log_metric("val_rmse", val_rmse)
            test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
            mlflow.log_metric("test_rmse", test_rmse)

    def register_model(self): 
        pass


if __name__ == '__main__':
    training_model = TrainingModel()
    training_model.train_model_and_track() 
        







        

