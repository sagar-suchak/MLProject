import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object

@dataclass
class data_transformation_config:
    preprocessor_obj_file_path= os.path.join('artifacts', 'preprocessor.pkl')
    logging.info('Pickle file loaded successfully')

class data_transformation:
    def __init__(self):
        self.data_transformation_config = data_transformation_config()

    def get_data_transformer(self):

        ### this function is responsible for Data Transformation

        try:
            numerical_features= ['reading_score', 'writing_score']
            cat_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("std_scaler", StandardScaler())

                ])
            cat_pipeline = Pipeline(

                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown='ignore',  sparse_output=False)),
                    ("std_scaler", StandardScaler(with_mean=False))
                ])

            logging.info('Catelogical and Numerical Pipeline created') 
            
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, cat_features)   

                ]
            )

            logging.info('Column Transformer created')

            return preprocessor
        
        except Exception as e:
            raise custom_exception(e,sys)
        


    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Train and test data read successfully')
        
            logging.info('Initiating Data Transformation')

            preprocessor_obj = self.get_data_transformer()    

            target_column = 'math_score' 
            numerical_features= ['reading_score', 'writing_score']    

            input_features_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]


            logging.info('Applying the preprocessor on the train and test data')

            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]

            logging.info('Data Transformation completed successfully')

            save_object(
            self.data_transformation_config.preprocessor_obj_file_path,
            preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise custom_exception(e,sys)
        