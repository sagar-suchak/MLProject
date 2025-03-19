import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import RandomizedSearchCV


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
    )
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object, evaluate_model
@dataclass
class model_trainer_config:
    trained_model_path = os.path.join('artifacts', 'model.pkl')
    logging.info('Pickle file loaded successfully')

class model_trainer:
    def __init__(self):
        self.model_trainer_config = model_trainer_config()
        

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Splitting the data into train and test data')

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            logging.info('Data split successfully')

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Rregression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }
        
            param_grid = {
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [5, 10, 15, None],
                },
                "Decision Tree": {
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2,5,10]
                },
                "Gradient Boosting" :{
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2]
                },
                "Linear Rregression": {}, #No parameteres to tune
                "K-Neighbours Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights":["uniform","distance"]
                },
                "XG Boost":{
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3,5,7]
                },
                "AdaBoost": {
                    "n_estimator":[50,100,200],
                    "learning_rate":[0.01,0.1,0.2, 1]
                },
                "Catboost": {
                    "iterations":[100, 200, 300],
                    "depth": [4,6,10],
                    "learning_rate":[0.01,0.1,0.2]
                }
            }

            best_model = None
            best_score = float('-inf')
            best_model_name = None

            for model_name, model in models.items():
                logging.info(f"Training {model_name} with hyperparameter tuning")

                if param_grid[model_name]:
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grid[model_name],
                        n_iter=10,
                        scoring = 'r2',
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        random_state=42
                    )
                    search.fit(X_train, y_train)
                    best_model_instance = search.best_estimator_
                else:
                    best_model_instance = model.fit(X_train, y_train)
                
                y_pred = best_model_instance.predict(X_test)
                score = r2_score(y_test, y_pred)

                logging.info(f"R2 Score for {model_name} is {score:.4f}")
                
                if score > best_score:
                        best_score = score
                        best_model = best_model_instance
                        best_model_name = model_name

                if best_score < 0.6:
                    raise custom_exception("No suitable model found. Best R² score is below 0.6.", sys)
            
                logging.info(f"Best model: {best_model_name} with R² Score: {best_score:.4f}")

                # Save the best model
                save_object(file_path = self.model_trainer_config.trained_model_path,obj =  best_model)
                logging.info("Best model saved successfully.")



                predicted = best_model.predict(X_test)
                
                r2_square = r2_score(y_test, predicted)
                return r2_square
            

        # model_report: dict = evaluate_model(X_train= X_train, y_train= y_train, X_test= X_test, y_test= y_test, models= models)

        # ## get the best model score form dictionary
        # best_model_score = max(sorted(model_report.values()))

        # ## get the best model name from dictionary 
        # best_model_name = list(model_report.keys())[
        #     list(model_report.values()).index(best_model_score)

        # ]

        
        except Exception as e:
            raise custom_exception(e,sys)
        