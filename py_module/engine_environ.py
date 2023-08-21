from py_module.config import Configuration
import gymnasium as gym

import enum
import numpy as np
import random
from collections import deque

LUBRICATION_LOOKBACK = 15
LUBRICATION_REWARD = -10
REPLACEMENT_REWARD = -250
DO_NOTHING_REWARD = 1
FAILURE_REWARD = -600
PREVIOUS_STATE_USED = Configuration().previous_p_times
ENGINE_AMOUNT = Configuration().train_engine_amount

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
        
    
    def reset(self, offset):

        assert offset >= self.previous_state_used - 1
        ### 重新抽取一支引擎資料
        self.unit = random.choice(ENGINE_AMOUNT) ### sample 1 engine #
        self.unit_data = self.engine_data[self.engine_data["unit"] == self.unit]
        print("Sample 1 engine data from source data: {}".format(self.unit))

        self._offset = offset
        self._data = self.unit_data
        self._cycle_num = len(self._data)

    
    @property # 只讀取
    def shape(self):
        ### 25個引擎特徵 * 批次狀態(15 + 1 states)
        return self.config_obj.features_num * self.previous_state_used
    
    def encode(self): ### Agent's observe
        """
        Convert current state into numpy array.
        """
        res = deque([], maxlen=self.previous_state_used)
        for state_cur in range(-self.previous_state_used+self._offset, self._offset):
            if state_cur < 0:
                res.append(self._data.iloc[0, ])
            else:
                res.append(self._data.iloc[state_cur, ])
        res = np.array(res)
        return res

    def step(self, action):
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        
        if action == Actions.Lubrication:
            self._offset -= LUBRICATION_LOOKBACK
            reward = LUBRICATION_REWARD
        elif action == Actions.Replacement:
            self._offset = 1
            reward = REPLACEMENT_REWARD
        else:
            self._offset += 1
            reward = DO_NOTHING_REWARD

        done |= self._offset > self._cycle_num ### 若offset > 最大cycle數，則done=True

        if done:
            reward = FAILURE_REWARD ### 故障發生，-600分
        return reward, done
    
class EngineEnv(gym.Env):

    metadata = {'render.modes':['human']}
    spec = gym.envs.registration.EnvSpec("EngineEnv-v0")

    def __init__(self, previous_state_used=PREVIOUS_STATE_USED, reward_on_EOL=True):

        self._state = State(previous_state_used, reward_on_EOL)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_shape = (self.config_obj.features_num) ### states shape, Engine contains 25 features
        self.observation_space = gym.spaces.Box(
            low = np.zeros(self.observation_shape),
            high = np.ones(self.observation_shape),
            dtype = np.float16
        )
    def reset(self):
        pass