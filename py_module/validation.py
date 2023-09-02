from py_module.torch_model import NoisyLinear, SimpleFFDQN, SimpleDNN, DQNConv1D, DQNConv1DLarge
from py_module.engine_environ import Actions, EngineEnv
from py_module.config import Configuration

import numpy as np

import torch

class ModelValidation(object):

    def __init__(self):
        self.METRICS = (
            'episode_reward',
            'episode_steps',
            'failure',
            'do_nothing_count',
            'lubrication_count',
            'replacement_count',
            'do_nothing_percent',
            'lubrication_percent',
            'replacement_percent',
        )
        self.config_obj = Configuration()


    def validation_run(self, data, args, episodes=1000, device="cpu"):
        stats = { metric: [] for metric in self.METRICS }

        env = EngineEnv(data)
        net = SimpleDNN(env.observation_space.shape[0], env.action_space.n, self.config_obj.previous_p_times).to(device)
        net.load_state_dict(torch.load(args.valid))

        for episode in range(episodes):
            
            obs = env.reset()

            total_reward = 0.0
            episode_steps = 0

            while True:
                obs_v = torch.tensor([obs]).to(device)
                out_v = net(obs_v)
                print("out_v", out_v)
                action_idx = out_v.max(dim=1)[1].item()
                print("action_idx", action_idx)
                # if np.random.random() < epsilon:
                #     action_idx = env.action_space.sample()

                obs, reward, done, truncated, _ = env.step(action_idx)
                total_reward += reward
                episode_steps += 1

            stats['episode_reward'].append(total_reward)
            stats['episode_steps'].append(episode_steps)

            if episode_steps < 1000:
                stats['failure'].append(True)
            else:
                stats['failure'].append(False)
            stats['do_nothing_count'].append(env.do_nothing_counter)
            stats['lubrication_count'].append(env.lubrication_counter)
            stats['replacement_count'].append(env.replacement_counter)

            stats['do_nothing_percent'].append(np.median(env.do_nothing_percent))
            stats['lubrication_percent'].append(np.median(env.lubrication_percent))
            stats['replacement_percent'].append(np.median(env.replacement_percent))


        return { key: np.mean(vals) for key, vals in stats.items() }
