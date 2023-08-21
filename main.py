from py_module.config import Configuration
from py_module.data_reader import DataReader
from py_module.data_preprocessing import DataProprocessing

import os
import pandas as pd

class MaintenanceStrategyRL(object):

    """
    Main Flow:
    1. data loading
    2. data preprocessing
    3. data training
    """

    # Preprocessing
    #   Define RUL
    #   Standardization
    # Feature Extraction
    #   AE
    # RUL Prediction

    def __init__(self):
        self.config_obj = Configuration()
        self.reader_obj = DataReader()
        self.data_preprocessing_obj = DataProprocessing()

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



def main_flow():
    
    main_obj = MaintenanceStrategyRL()
    
    data, testing_data = main_obj.data_loading()
    data = main_obj.data_preprocessing(data)
    testing_data = main_obj.data_preprocessing(testing_data)

    unit_data = data[data['unit'] == 50]
    print(unit_data)
    print(unit_data.iloc[500,])
    # Training


if __name__ == "__main__":
    main_flow()

