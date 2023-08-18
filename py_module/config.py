import os
import time

class Configuration(object):
    
    def __init__(self):

        self.data_folder = os.path.join("datasets", "Data_2008_PHM")
        self.file_name = "train.txt"

        self.features_name = ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',] + ['sensor_' + str(i) for i in range(1, 24)]
        self.features_num = 25

        self.test_data_folder = os.path.join("datasets", "Data_2008_PHM")
        self.test_file_name = "final_test.txt"

        # 2008 Engine Data
        self.train_engine_number = 218
        self.test_engine_number = 218
        self.standardization_features = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + ['sensor_' + str(i) for i in range(1, 22)]
        self.previous_p_times = 15
        self.checkpoint_path = os.path.join(self.data_folder, "training_record", "cp.ckpt")
        self.keras_model_path = os.path.join(self.data_folder, "training_record", "keras_model", "{}_model.h5".format(time.ctime().split()[0:3]))
        # self.checkpoint_path = self.data_folder + "\\training_record\\cp.ckpt"
        # self.keras_model_path = self.data_folder + "\\training_record\\keras_model\\109-09-06_model.h5"
        self.keras_updated_model_path = os.path.join(self.data_folder, "training_record","keras_model","{}_model.h5".format(time.ctime().split()[0:3]))
