import pandas as pd

class DataReader(object):

    def __init__(self):
        pass

    def read_csv_data(self, data_path):
        
        data = pd.read_csv(data_path, sep=' ', header=None)

        return data