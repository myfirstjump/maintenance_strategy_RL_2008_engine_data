from py_module.config import Configuration

import numpy as np
import gymnasium as gym
import random
from collections import deque

class EngineEquipment(object):
    def __init__(self, source_data):
        self.config_obj = Configuration()
        self.unit = random.choice(self.config_obj.train_engine_number) ### sample 1 engine# from training set
        self.unit_data = source_data[source_data["unit"] == self.unit]
        print("Sample 1 engine data from source data: {}".format(self.unit))
        self.record_cursor = 0
        self.ob = self.unit_data.iloc[self.record_cursor, ]
        print('New ob is {}'.format(self.ob))
    
    def nothing_just_degrade(self):
        pass
    def be_lubricated(self):
        pass
    def be_replaced(self):
        pass

class EngineStrategy(gym.Env):
    def __init__(self, data, k):
        super(EngineStrategy, self).__init__()
        self.config_obj = Configuration()

        self.k = k ### K表示加入歷史前K個狀態成為「批次狀態」
        self.frames = deque([], maxlen=k)

        self.observation_shape = (self.config_obj.features_num) ### states shape, Engine contains 25 features
        self.observation_space = gym.spaces.Box(
            low = np.zeros(self.observation_shape),
            high = np.ones(self.observation_shape),
            dtype = np.float16
        )

        self.action_space = gym.spaces.Discrete(3,) ### A1: no action, A2: lubrication, A3: replacement

        self.episode_max_cycle = 1000
    
    def _get_ob(self):
        return np.array(self.frames)

    def reset(self):
        """
        Resets the environment to its initial state and returns the initial observation.
        """

        self.episode_cnt = 0 ### 計數器，用來確認是否達到episode_max_cycle = 1000
        self.episode_reward = 0
        self.engine_data = EngineEquipment()

        ### 加入先前k次狀態(批次狀態)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def get_action_meanings(self):
        return {0:"no action", 1:"lubrication", 2:"replacement"}

    def step(self, action):
        """
        Executes a step in the environment by applying an action. Returns the new observation, reward, completion status, and other info.
        """
        # Flag that marks the termination of an episode
        done = False

        assert self.action_space.contains(action), "Invalid Action"

        self.episode_cnt += 1

        reward = 1 # 裝備持續正常服役

        ### apply the action to the Engine
        if action == 0: ### no action
            print("No action, just normal degrade...")
        elif action == 1:
            print("Do some lubricate to the engine!")
            reward = -5
        elif action == 2:
            print("Directly replace another new engine!")
            reward = -50

        # # Increment the episodic return
        # self.episode_reward += 1

        if self.episode_cnt >= self.episode_max_cycle: ### 若服役達1000 cycles，則視為回合結束
            done = True

        return self.frames, reward, done, []
    

    env = EngineEquipment()
    
        


