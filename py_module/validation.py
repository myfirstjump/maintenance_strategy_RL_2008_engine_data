from py_module.torch_model import NoisyLinear, SimpleFFDQN, SimpleDNN, DQNConv1D, DQNConv1DLarge, SimpleDNN_small
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
            'Replacement_Left_cycle'
        )
        self.config_obj = Configuration()


    def validation_run(self, data, args, episodes=20, device="cpu"):
        stats = { metric: [] for metric in self.METRICS }

        env = EngineEnv(data)
        net = SimpleDNN_small(env.observation_space.shape[0], env.action_space.n, self.config_obj.previous_p_times).to(device)
        net.load_state_dict(torch.load(args.valid))

        for episode in range(episodes):
            print('Episode: {}'.format(episode))
            obs = env.reset()

            total_reward = 0.0
            episode_steps = 0

            while True:
                state_a = np.array([obs], copy=False)
                state_v = torch.tensor(state_a).to(device).view((self.config_obj.features_num * self.config_obj.previous_p_times))
                q_vals_v = net(state_v)
                # print("out_v", q_vals_v)
                _, act_v = torch.max(q_vals_v, dim=0)
                action = int(act_v.item())
                # print("action_idx", action)
                # if np.random.random() < epsilon:
                #     action_idx = env.action_space.sample()

                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                episode_steps += 1

                if done:
                    break

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

            stats['Replacement_Left_cycle'].append(env.Replacement_Left_cycle)


        return { key: np.mean(vals) for key, vals in stats.items() }
