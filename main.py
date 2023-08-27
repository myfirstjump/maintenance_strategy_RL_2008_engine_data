from py_module.config import Configuration
from py_module.data_reader import DataReader
from py_module.data_preprocessing import DataProprocessing
from py_module.dqn_train import RLModeTrian

import argparse
import os
import pandas as pd

class MaintenanceStrategyRL(object):

    """
    Main Flow:
    1. data loading
    2. data preprocessing
    3. data training
    """

    def __init__(self):
        self.config_obj = Configuration()
        self.reader_obj = DataReader()
        self.data_preprocessing_obj = DataProprocessing()
        self.train_obj = RLModeTrian()

    def data_loading(self):
        file_path = os.path.join(self.config_obj.data_folder, self.config_obj.file_name)
        data = self.reader_obj.read_csv_data(file_path)
        

        test_file_path = os.path.join(self.config_obj.test_data_folder, self.config_obj.test_file_name)
        testing_data = self.reader_obj.read_csv_data(test_file_path)
        return data, testing_data

    def data_preprocessing(self, data):
        
        data = self.data_preprocessing_obj.data_preprocessing_2008_PHM_Engine_data(data, self.config_obj.features_name)
        # data = self.data_preprocessing_obj.features_standardization(data, self.config_obj.standardization_features)

        return data
    
    def model_train(self, data, val_data, args):

        self.train_obj.model_training(data, val_data, args)




def main_flow(args):
    
    main_obj = MaintenanceStrategyRL()
    
    data, testing_data = main_obj.data_loading()
    data = main_obj.data_preprocessing(data)
    testing_data = main_obj.data_preprocessing(testing_data)

    # Training
    main_obj.model_train(data, testing_data, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", required=True, help="Run name")
    args = parser.parse_args()

    main_flow(args)

