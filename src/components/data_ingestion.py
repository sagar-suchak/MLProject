import os
import sys
from src.exception import custom_exception
from src.logger import logging
from src.components.data_transformation import data_transformation
from src.components.data_transformation import data_transformation_config
from src.components.model_trainer import model_trainer
from src.components.model_trainer import model_trainer_config


import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class data_ingestion_config:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class data_ingestion:

    def __init__(self):
        self.ingestion_config = data_ingestion_config()

    def initiate_data_ingestion(self):
        logging.info('Entered the Data Ingestion Module')
        try: 
            df = pd.read_csv('notebook\data\Student_data.csv')
            logging.info('Read the dataset successfully')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) 

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) 

            logging.info('Train test split initiated')

            train_set, test_set =  train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) 

            logging.info('Ingestion of the data completed successfully')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:

            raise custom_exception(e,sys)


if __name__ == "__main__":
    obj=  data_ingestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation_obj = data_transformation()
    train_arr, test_arr, _ = data_transformation_obj.initiate_data_transformation(train_data, test_data)

    model_trainer = model_trainer()
    print(model_trainer.initiate_model_training(train_arr, test_arr))

