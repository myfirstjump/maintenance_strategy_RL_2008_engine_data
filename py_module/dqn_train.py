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


Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class RLModeTrian(object):
    def __init__(self):
        pass
    
    # def model_training(self, data, val_data, args):

    #     saves_path = SAVES_DIR / f"simple-{args.run}"
    #     env = EngineEnv(data)
    #     env_tst = EngineEnv(data)
    #     env_val = EngineEnv(val_data)
        
    #     print("env.action_space.n", env.action_space.n)
    #     net = SimpleFFDQN(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    #     # net = SimpleDNN(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    #     tgt_net = ptan.agent.TargetNet(net)

    #     selector = ptan.actions.EpsilonGreedyActionSelector(EPS_START) ### 行動選擇器

    #     eps_tracker = ptan.actions.EpsilonTracker(selector, EPS_START, EPS_FINAL, EPS_STEPS)
    #     agent = ptan.agent.DQNAgent(net, selector, device=DEVICE)
    #     exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    #     buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
    #     optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    #     def process_batch(engine, batch):
    #         optimizer.zero_grad()
    #         loss_v = common.calc_loss(
    #             batch, net, tgt_net.target_model,
    #             gamma=GAMMA ** REWARD_STEPS, device=DEVICE)
    #         loss_v.backward()
    #         optimizer.step()
    #         eps_tracker.frame(engine.state.iteration)

    #         if getattr(engine.state, "eval_states", None) is None:
    #             eval_states = buffer.sample(STATES_TO_EVALUATE)
    #             eval_states = [np.array(transition.state, copy=False)
    #                         for transition in eval_states]
    #             engine.state.eval_states = np.array(eval_states, copy=False)

    #         return {
    #             "loss": loss_v.item(),
    #             "epsilon": selector.epsilon,
    #         }
    
    #     engine = Engine(process_batch)
    #     tb = common.setup_ignite(engine, exp_source, f"simple-{args.run}",
    #                          extra_metrics=('values_mean',))
    
    #     @engine.on(ptan.ignite.PeriodEvents.ITERS_1000_COMPLETED)
    #     def sync_eval(engine: Engine):
    #         tgt_net.sync()

    #         mean_val = common.calc_values_of_states(
    #             engine.state.eval_states, net, device=DEVICE)
    #         engine.state.metrics["values_mean"] = mean_val
    #         if getattr(engine.state, "best_mean_val", None) is None:
    #             engine.state.best_mean_val = mean_val
    #         if engine.state.best_mean_val < mean_val:
    #             print("{}: Best mean value updated {} -> {}".format(
    #                 engine.state.iteration, engine.state.best_mean_val,
    #                 mean_val))
    #             path = saves_path / ("mean_value-{}.data".format(mean_val))
    #             torch.save(net.state_dict(), path)
    #             engine.state.best_mean_val = mean_val

    #     @engine.on(ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED)
    #     def validate(engine: Engine):
    #         res = validation.validation_run(env_tst, net, device=DEVICE)
    #         print("{}: tst: {}".format(engine.state.iteration, res))
    #         for key, val in res.items():
    #             engine.state.metrics[key + "_tst"] = val
    #         res = validation.validation_run(env_val, net, device=DEVICE)
    #         print("{}: val: {}".format(engine.state.iteration, res))
    #         for key, val in res.items():
    #             engine.state.metrics[key + "_val"] = val
    #         val_reward = res['episode_reward']
    #         if getattr(engine.state, "best_val_reward", None) is None:
    #             engine.state.best_val_reward = val_reward
    #         if engine.state.best_val_reward < val_reward:
    #             print("Best validation reward updated: {} -> {}, model saved".format(
    #                 engine.state.best_val_reward, val_reward
    #             ))
    #             engine.state.best_val_reward = val_reward
    #             path = saves_path / ("val_reward-{}.data".format(val_reward))
    #             torch.save(net.state_dict(), path)
        
    #     event = ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED
    #     tst_metrics = [m + "_tst" for m in validation.METRICS]
    #     tst_handler = tb_logger.OutputHandler(
    #         tag="test", metric_names=tst_metrics)
    #     tb.attach(engine, log_handler=tst_handler, event_name=event)

    #     val_metrics = [m + "_val" for m in validation.METRICS]
    #     val_handler = tb_logger.OutputHandler(
    #         tag="validation", metric_names=val_metrics)
    #     tb.attach(engine, log_handler=val_handler, event_name=event)
    #     engine.run(common.batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE))

    def model_training(self, data, val_data, args):

        env = EngineEnv(data)
        net = SimpleDNN(env.observation_space.shape[0], env.action_space.n, config_obj.previous_p_times).to(DEVICE)
        # net = SimpleFFDQN(env.observation_space.shape[0]).to(DEVICE)

        ### target net用於生成網路比較用的target --> Q(s,a) = r + Q(t+1)(s, a)
        tgt_net = SimpleDNN(env.observation_space.shape[0], env.action_space.n, config_obj.previous_p_times).to(DEVICE)
        # tgt_net = SimpleFFDQN(env.observation_space.shape[0]).to(DEVICE)
        writer = SummaryWriter(comment="-" + args.run)
        print(net)

        buffer = ExperienceBuffer(REPLAY_SIZE)
        agent = Agent(env, buffer)
        epsilon = EPSILON_START

        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        total_rewards = []
        frame_idx = 0
        ts_frame = 0
        ts = time.time()
        best_m_reward = None

        while True:
            frame_idx += 1
            
            
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

            reward, action_counter_dict = agent.play_step(net, epsilon, device=DEVICE)
            # if frame_idx % 10000 == 0:
            #     print(" === === === Training episode: {} === === ===".format(frame_idx))
            #     # print("total_rewards", total_rewards)
            if reward is not None:
                total_rewards.append(reward)
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                m_reward = np.mean(total_rewards[-100:])
                print("{}: done {} games, reward {}, eps {}, speed {} f/s".format(frame_idx, len(total_rewards), m_reward, epsilon, speed))
                
                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("reward_100", m_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)

                writer.add_scalar("Steps", action_counter_dict['Step'], frame_idx)
                writer.add_scalar("Do_nothing", action_counter_dict['Do_nothing'], frame_idx)
                writer.add_scalar("Lubrication", action_counter_dict['Lubrication'], frame_idx)
                writer.add_scalar("Replacement", action_counter_dict['Replacement'], frame_idx)

                try:
                    writer.add_histogram("Do_nothing Histogram during Degradation", action_counter_dict['Do_nothing_%'], bins='auto')
                    writer.add_histogram("Lubrication Histogram during Degradation", action_counter_dict['Lubrication_%'], bins='auto')
                    writer.add_histogram("Replacement Histogram during Degradation", action_counter_dict['Replacement_%'], bins='auto')

                except:
                    pass
                if best_m_reward is None or best_m_reward < m_reward:
                    # torch.save(net.state_dict(), ENV_NAME +
                    #         "-best_%.0f.dat" % m_reward)
                    if best_m_reward is not None:
                        print("Best reward updated {} -> {}".format(best_m_reward, m_reward))
                    best_m_reward = m_reward
                if m_reward > MEAN_REWARD_BOUND:
                    print("Solved in {} frames!".format(frame_idx))
                    try:
                        writer.add_histogram("Do_nothing Histogram during Degradation last 1000", action_counter_dict['Do_nothing_%'][-1000:], bins='auto')
                        writer.add_histogram("Lubrication Histogram during Degradation last 1000", action_counter_dict['Lubrication_%'][-1000:], bins='auto')
                        writer.add_histogram("Replacement Histogram during Degradation last 1000", action_counter_dict['Replacement_%'][-1000:], bins='auto')
                    except:
                        pass
                    break

            if len(buffer) < REPLAY_START_SIZE:
                continue

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=DEVICE)
            loss_t.backward()
            optimizer.step()
        writer.close()

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        action_counter_dict = {'Step':None, "Do_nothing":None, "Lubrication":None, "Replacement":None, "Do_nothing_%":None, "Lubrication_%":None, "Replacement_%":None,}

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # print("In agent:")
            state_a = np.array([self.state], copy=False)
            # print("state_a", state_a, type(state_a))
            state_v = torch.tensor(state_a).to(device).view((config_obj.features_num * config_obj.previous_p_times))
            
            q_vals_v = net(state_v)
            # print("q_vals_v ", q_vals_v)
            _, act_v = torch.max(q_vals_v, dim=0)
            action = int(act_v.item())
            # print("act_v, action", act_v, action)

        # do step in the environment
        new_state, reward, is_done, _, info = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            print("Step count: {}, do_nothing count: {}, lubrication count: {}, replacment count: {}".format(self.env.step_counter, self.env.do_nothing_counter, self.env.lubrication_counter, self.env.replacement_counter))
            action_counter_dict['Step'] = self.env.step_counter
            action_counter_dict['Do_nothing'] = self.env.do_nothing_counter
            action_counter_dict['Lubrication'] = self.env.lubrication_counter
            action_counter_dict['Replacement'] = self.env.replacement_counter

            action_counter_dict['Do_nothing_%'] = self.env.do_nothing_percent
            action_counter_dict['Lubrication_%'] = self.env.lubrication_percent
            action_counter_dict['Replacement_%'] = self.env.replacement_percent
            
            done_reward = self.total_reward
            self._reset()
        return done_reward, action_counter_dict


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_vec = torch.tensor(np.array(states, copy=False)).to(device).view((BATCH_SIZE, -1))
    next_states_vec = torch.tensor(np.array(next_states, copy=False)).to(device).view((BATCH_SIZE, -1))
    actions_vec = torch.tensor(actions).to(device)
    rewards_vec = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # print("states_vec", states_vec, states_vec.shape)
    # print("next_states_vec", next_states_vec, next_states_vec.shape)
    # print("actions_vec", actions_vec, actions_vec.shape)
    # print("rewards_vec", rewards_vec, rewards_vec.shape)
    # print("done_mask", done_mask, done_mask.shape)

    state_action_values = net(states_vec).gather(1, actions_vec.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_vec).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + \
                                   rewards_vec
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)



