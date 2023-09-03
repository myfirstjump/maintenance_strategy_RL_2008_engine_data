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
        self.train_engine_amount = 218
        self.test_engine_amount = 218
        self.standardization_features = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + ['sensor_' + str(i) for i in range(1, 22)]
        self.checkpoint_path = os.path.join(self.data_folder, "training_record", "cp.ckpt")
        self.keras_model_path = os.path.join(self.data_folder, "training_record", "keras_model", "{}_model.h5".format(time.ctime().split()[0:3]))
        # self.checkpoint_path = self.data_folder + "\\training_record\\cp.ckpt"
        # self.keras_model_path = self.data_folder + "\\training_record\\keras_model\\109-09-06_model.h5"
        self.keras_updated_model_path = os.path.join(self.data_folder, "training_record","keras_model","{}_model.h5".format(time.ctime().split()[0:3]))

        
        
        ### RL environment settings
        self.LUBRICATION_LOOKBACK = 10
        self.LUBRICATION_REWARD = -5
        self.REPLACEMENT_REWARD = -100
        self.DO_NOTHING_REWARD = 1
        self.FAILURE_REWARD = -5000

        ### RL strategy settings
        self.DEVICE = 'cpu'
        self.MEAN_REWARD_BOUND = 100
        self.previous_p_times = 5 ### RL states

        self.GAMMA = 0.99
        self.BATCH_SIZE = 32
        self.REPLAY_SIZE = 10000
        self.LEARNING_RATE = 1e-4
        self.SYNC_TARGET_FRAMES = 10000
        self.REPLAY_START_SIZE = 10000

        self.EPSILON_DECAY_LAST_FRAME = 500000
        self.EPSILON_START = 1.0
        self.EPSILON_FINAL = 0.01
