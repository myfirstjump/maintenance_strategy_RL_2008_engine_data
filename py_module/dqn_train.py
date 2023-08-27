from py_module.torch_model import NoisyLinear, SimpleFFDQN, SimpleDNN, DQNConv1D, DQNConv1DLarge
from py_module.engine_environ import Actions, State, EngineEnv
from py_module.config import Configuration
from py_module import common, validation

import time
import pathlib
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import ptan

from ignite.engine import Engine
from ignite.contrib.handlers import tensorboard_logger as tb_logger

from tensorboardX import SummaryWriter

config_obj = Configuration()

ENV_NAME = "customized-MS-env"
DEVICE = config_obj.DEVICE
MEAN_REWARD_BOUND = config_obj.MEAN_REWARD_BOUND

GAMMA = config_obj.GAMMA
BATCH_SIZE = config_obj.BATCH_SIZE
REPLAY_SIZE = config_obj.REPLAY_SIZE
LEARNING_RATE = config_obj.LEARNING_RATE
SYNC_TARGET_FRAMES = config_obj.SYNC_TARGET_FRAMES
REPLAY_START_SIZE = config_obj.REPLAY_START_SIZE

EPSILON_DECAY_LAST_FRAME = config_obj.EPSILON_DECAY_LAST_FRAME
EPSILON_START = config_obj.EPSILON_START
EPSILON_FINAL = config_obj.EPSILON_FINAL


### TBD
SAVES_DIR = pathlib.Path("saves")
EPS_START = 1.0
EPS_FINAL = 0.1
EPS_STEPS = 1000000

REWARD_STEPS = 1000
STATES_TO_EVALUATE = 250

REPLAY_INITIAL = 500

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class RLModeTrian(object):
    def __init__(self):
        pass
    
    def model_training(self, data, val_data, args):

        saves_path = SAVES_DIR / f"simple-{args.run}"
        env = EngineEnv(data)
        env_tst = EngineEnv(data)
        env_val = EngineEnv(val_data)
        
        print("env.action_space.n", env.action_space.n)
        net = SimpleFFDQN(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
        # net = SimpleDNN(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
        tgt_net = ptan.agent.TargetNet(net)

        selector = ptan.actions.EpsilonGreedyActionSelector(EPS_START) ### 行動選擇器

        eps_tracker = ptan.actions.EpsilonTracker(selector, EPS_START, EPS_FINAL, EPS_STEPS)
        agent = ptan.agent.DQNAgent(net, selector, device=DEVICE)
        exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
        buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
        def process_batch(engine, batch):
            optimizer.zero_grad()
            loss_v = common.calc_loss(
                batch, net, tgt_net.target_model,
                gamma=GAMMA ** REWARD_STEPS, device=DEVICE)
            loss_v.backward()
            optimizer.step()
            eps_tracker.frame(engine.state.iteration)

            if getattr(engine.state, "eval_states", None) is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False)
                            for transition in eval_states]
                engine.state.eval_states = np.array(eval_states, copy=False)

            return {
                "loss": loss_v.item(),
                "epsilon": selector.epsilon,
            }
    
        engine = Engine(process_batch)
        tb = common.setup_ignite(engine, exp_source, f"simple-{args.run}",
                             extra_metrics=('values_mean',))
    
        @engine.on(ptan.ignite.PeriodEvents.ITERS_1000_COMPLETED)
        def sync_eval(engine: Engine):
            tgt_net.sync()

            mean_val = common.calc_values_of_states(
                engine.state.eval_states, net, device=DEVICE)
            engine.state.metrics["values_mean"] = mean_val
            if getattr(engine.state, "best_mean_val", None) is None:
                engine.state.best_mean_val = mean_val
            if engine.state.best_mean_val < mean_val:
                print("{}: Best mean value updated {} -> {}".format(
                    engine.state.iteration, engine.state.best_mean_val,
                    mean_val))
                path = saves_path / ("mean_value-{}.data".format(mean_val))
                torch.save(net.state_dict(), path)
                engine.state.best_mean_val = mean_val

        @engine.on(ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED)
        def validate(engine: Engine):
            res = validation.validation_run(env_tst, net, device=DEVICE)
            print("{}: tst: {}".format(engine.state.iteration, res))
            for key, val in res.items():
                engine.state.metrics[key + "_tst"] = val
            res = validation.validation_run(env_val, net, device=DEVICE)
            print("{}: val: {}".format(engine.state.iteration, res))
            for key, val in res.items():
                engine.state.metrics[key + "_val"] = val
            val_reward = res['episode_reward']
            if getattr(engine.state, "best_val_reward", None) is None:
                engine.state.best_val_reward = val_reward
            if engine.state.best_val_reward < val_reward:
                print("Best validation reward updated: {} -> {}, model saved".format(
                    engine.state.best_val_reward, val_reward
                ))
                engine.state.best_val_reward = val_reward
                path = saves_path / ("val_reward-{}.data".format(val_reward))
                torch.save(net.state_dict(), path)
        
        event = ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED
        tst_metrics = [m + "_tst" for m in validation.METRICS]
        tst_handler = tb_logger.OutputHandler(
            tag="test", metric_names=tst_metrics)
        tb.attach(engine, log_handler=tst_handler, event_name=event)

        val_metrics = [m + "_val" for m in validation.METRICS]
        val_handler = tb_logger.OutputHandler(
            tag="validation", metric_names=val_metrics)
        tb.attach(engine, log_handler=val_handler, event_name=event)
        engine.run(common.batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE))

    # def model_training(self, data):

    #     env = EngineEnv(data)
    #     net = SimpleDNN(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    #     # net = SimpleDNN(env.observation_space.shape[0]).to(DEVICE)
    #     tgt_net = SimpleDNN(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    #     # tgt_net = SimpleDNN(env.observation_space.shape[0]).to(DEVICE)
    #     writer = SummaryWriter(comment="-" + ENV_NAME)
    #     print(net)

    #     buffer = ExperienceBuffer(REPLAY_SIZE)
    #     agent = Agent(env, buffer)
    #     epsilon = EPSILON_START

    #     optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    #     total_rewards = []
    #     frame_idx = 0
    #     ts_frame = 0
    #     ts = time.time()
    #     best_m_reward = None

    #     while True:
    #         frame_idx += 1
    #         if frame_idx % 100 == 0:
    #             print(" === === === Training episode: {} === === ===".format(frame_idx))
            
    #         epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

    #         reward = agent.play_step(net, epsilon, device=DEVICE)
    #         if reward is not None:
    #             total_rewards.append(reward)
    #             speed = (frame_idx - ts_frame) / (time.time() - ts)
    #             ts_frame = frame_idx
    #             ts = time.time()
    #             m_reward = np.mean(total_rewards[-100:])
    #             print("{}: done {} games, reward {}, eps {}, speed {} f/s".format(frame_idx, len(total_rewards), m_reward, epsilon, speed))
    #             writer.add_scalar("epsilon", epsilon, frame_idx)
    #             writer.add_scalar("speed", speed, frame_idx)
    #             writer.add_scalar("reward_100", m_reward, frame_idx)
    #             writer.add_scalar("reward", reward, frame_idx)
    #             if best_m_reward is None or best_m_reward < m_reward:
    #                 torch.save(net.state_dict(), ENV_NAME +
    #                         "-best_%.0f.dat" % m_reward)
    #                 if best_m_reward is not None:
    #                     print("Best reward updated {} -> {}".format(best_m_reward, m_reward))
    #                 best_m_reward = m_reward
    #             if m_reward > MEAN_REWARD_BOUND:
    #                 print("Solved in {} frames!".format(frame_idx))
    #                 break

    #         if len(buffer) < REPLAY_START_SIZE:
    #             continue

    #         if frame_idx % SYNC_TARGET_FRAMES == 0:
    #             tgt_net.load_state_dict(net.state_dict())

    #         optimizer.zero_grad()
    #         batch = buffer.sample(BATCH_SIZE)
    #         loss_t = calc_loss(batch, net, tgt_net, device=DEVICE)
    #         loss_t.backward()
    #         optimizer.step()
    #     writer.close()

# class ExperienceBuffer:
#     def __init__(self, capacity):
#         self.buffer = collections.deque(maxlen=capacity)

#     def __len__(self):
#         return len(self.buffer)

#     def append(self, experience):
#         self.buffer.append(experience)

#     def sample(self, batch_size):
#         indices = np.random.choice(len(self.buffer), batch_size, replace=False)
#         states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
#         return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)

# class Agent:
#     def __init__(self, env, exp_buffer):
#         self.env = env
#         self.exp_buffer = exp_buffer
#         self._reset()

#     def _reset(self):
#         self.state = self.env.reset()
#         self.total_reward = 0.0

#     @torch.no_grad()
#     def play_step(self, net, epsilon=0.0, device="cpu"):
#         done_reward = None

#         if np.random.random() < epsilon:
#             action = self.env.action_space.sample()
#         else:
#             print("In agent:")
#             state_a = np.array([self.state], copy=False)
#             print("state_a", state_a, type(state_a))
#             state_v = torch.tensor(state_a).to(device)
#             print("state_v", state_v, type(state_v))
#             q_vals_v = net(state_v)
#             _, act_v = torch.max(q_vals_v, dim=1)
#             action = int(act_v.item())

#         # do step in the environment
#         new_state, reward, is_done, _, info = self.env.step(action)
#         self.total_reward += reward

#         exp = Experience(self.state, action, reward,
#                          is_done, new_state)
#         self.exp_buffer.append(exp)
#         self.state = new_state
#         if is_done:
#             done_reward = self.total_reward
#             self._reset()
#         return done_reward


# def calc_loss(batch, net, tgt_net, device="cpu"):
#     states, actions, rewards, dones, next_states = batch

#     states_vec = torch.tensor(np.array(states, copy=False)).to(device)
#     next_states_vec = torch.tensor(np.array(next_states, copy=False)).to(device)
#     actions_vec = torch.tensor(actions).to(device)
#     rewards_vec = torch.tensor(rewards).to(device)
#     done_mask = torch.BoolTensor(dones).to(device)

#     print("states_vec", states_vec, states_vec.shape)
#     print("next_states_vec", next_states_vec, next_states_vec.shape)
#     print("actions_vec", actions_vec, actions_vec.shape)
#     print("rewards_vec", rewards_vec, rewards_vec.shape)
#     print("done_mask", done_mask, done_mask.shape)


#     tmp = net(states_vec)
#     print(tmp)


#     state_action_values = net(states_vec).gather(1, actions_vec.unsqueeze(-1)).squeeze(-1)
#     with torch.no_grad():
#         next_state_values = tgt_net(next_states_vec).max(1)[0]
#         next_state_values[done_mask] = 0.0
#         next_state_values = next_state_values.detach()

#     expected_state_action_values = next_state_values * GAMMA + \
#                                    rewards_vec
#     return nn.MSELoss()(state_action_values,
#                         expected_state_action_values)



