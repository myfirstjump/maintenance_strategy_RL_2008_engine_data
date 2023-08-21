from py_module.config import Configuration
import gymnasium as gym

import enum
import numpy as np
import random
from collections import deque

class Actions(enum.Enum):
    Nothing = 0
    Lubrication = 1
    Replacement = 2

class State:
    def __init__(self, source_data, previous_state_used, reward_on_EOL=True):
        
        assert isinstance(previous_state_used, int)
        assert previous_state_used > 0
        assert isinstance(reward_on_EOL, bool)
        self.previous_state_used = previous_state_used
        self.reward_on_EOL = reward_on_EOL

        ### 導入引擎資料
        self.engine_data = source_data
        self.config_obj = Configuration()
    
    def reset(self, offset):

        assert offset >= self.previous_state_used - 1
        ### 重新抽取一支引擎資料
        self.unit = random.choice(self.config_obj.train_engine_number) ### sample 1 engine #
        self.unit_data = self.engine_data[self.engine_data["unit"] == self.unit]
        print("Sample 1 engine data from source data: {}".format(self.unit))

        self._offset = offset
        self._data = self.unit_data
        self._cycle_num = len(self._data)

        return 
    
    @property # 只讀取
    def shape(self):
        ### 25個引擎特徵 * 批次狀態(15 + 1 states)
        return self.config_obj.features_num * self.previous_state_used
    
    def encode(self): ### Agent's observe
        """
        Convert current state into numpy array.
        """
        res = deque([], maxlen=self.previous_state_used)
        for state_cur in range(-self.previous_state_used+1, 1):
            res.append(self._data.iloc[state_cur, ])
        
        return 

