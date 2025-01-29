from torch import nn
import torch
import gymnasium as gym
from collections import deque, namedtuple
import itertools
import numpy as np
import random
import math
import ale_py
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, 
    MaxAndSkipEnv, 
    EpisodicLifeEnv,
    FireResetEnv
)
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

TOTAL_STEPS = int(5e6)
GAMMA = 0.99
BATCH_SIZE = 32
TOTAL_REPLAY_SIZE = int(1e6)
MIN_REPLAY_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = int(1e6)
NUM_ENVS = 4
TARGET_UPDATE_FREQ = 10000 // NUM_ENVS
LR = 6.25e-5 # scaled down by a factor of 4
# Hyperparameters for prioritized experience replay
PRIORITY_SCALE = 0.6
BETA_START = 0.4
BETA_END = 1.0
BETA_DECAY = int(5e6)

LOG_DIR = "./logs/Priotitized_DDQN" + str(LR)
LOG_INTERVAL = 1000
SAVE_DIR = "./models/Priotitized_DDQN"
SAVE_INTERVAL = 10000

gym.register_envs(ale_py)


def make_env():
    # Preprocessing only works with v4-NoFrameskip environments
    env = gym.make("SpaceInvadersNoFrameskip-v4")
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    # take action on reset for environments that are fixed until fire is pressed
    env = FireResetEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    return env


envs = gym.vector.SyncVectorEnv([lambda: Monitor(make_env(), allow_early_resets=True) for _ in range(NUM_ENVS)])


class Network(nn.Module):
    def __init__(self, env, device):
        super().__init__()

        self.num_actions = env.single_action_space.n
        self.device = device
        n_inputs = env.single_observation_space.shape[0]

        pre_cnn = nn.Sequential(
        nn.Conv2d(n_inputs, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten())

        with torch.no_grad():
            n_flatten = pre_cnn(torch.as_tensor(env.single_observation_space.sample()[None]).float()).shape[1]

        post_cnn = nn.Sequential(pre_cnn, nn.Linear(n_flatten, 512), nn.ReLU())

        self.net = nn.Sequential(post_cnn, nn.Linear(512, self.num_actions))

    def forward(self, x):
        return self.net(x / 255)

    def act(self, states, epsilon):
        states_t = torch.Tensor(states).to(self.device)
        q_values = self(states_t)

        actions = torch.argmax(q_values, dim=1).cpu().numpy()

        for i in range(len(actions)):
            random_sample = random.random()
            if random_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions - 1)

        return actions


class SummedTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.pointer = 0
        self.current_length = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, idx, value):
        node_index = idx + self.size - 1
        difference = value - self.nodes[node_index]

        self.nodes[node_index] = value

        parent_idx = (node_index - 1) // 2
        while parent_idx >= 0:
            self.nodes[parent_idx] += difference
            parent_idx = (parent_idx - 1) // 2

    def add(self, value, element):
        self.data[self.pointer] = element
        self.update(self.pointer, value)

        self.pointer = (self.pointer + 1) % self.size
        self.current_length = min(self.size, self.current_length + 1)

    def find(self, target):

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left_child, right_child = 2 * idx + 1, 2 * idx + 2

            if target <= self.nodes[left_child]:
                idx = left_child
            else:
                target -= self.nodes[left_child]
                idx = right_child

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'next_state'))
# Proportional Prioritized Experience Replay changes here
class PrioritizedReplayMemory:
    def __init__(self, buffer_size, eps=1e-2, alpha=0.1):
        self.tree = SummedTree(buffer_size)

        # PER params
        self.eps = eps 
        self.alpha = alpha  
        self.max_priority = eps  

        # transition: state, action, reward, next_state, done
        self.buffer = np.empty(buffer_size, dtype=Transition)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        state, action, reward, done, next_state = transition

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.buffer[self.count] = Transition(state, action, reward, done, next_state)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size, beta):

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)

            tree_idx, priority, sample_idx = self.tree.find(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total

        weights = (self.real_size * probs) ** -beta

        # normalize weights
        weights = weights / weights.max()

        batch = self.buffer[sample_idxs]
            
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

replay_memory = PrioritizedReplayMemory(TOTAL_REPLAY_SIZE, eps=1e-2, alpha=PRIORITY_SCALE)
episodeinfos_buffer = deque([], maxlen=100)

episode_count = 0
summary_writer = SummaryWriter(LOG_DIR)

Q_net = Network(envs, device=device)
target_net = Network(envs, device=device)

Q_net = Q_net.to(device)
target_net = target_net.to(device)

target_net.load_state_dict(Q_net.state_dict())

optimizer = torch.optim.Adam(Q_net.parameters(), lr=LR)

# Initialize replay buffer
states = envs.reset()[0]
for i in range(MIN_REPLAY_SIZE):
    actions = np.array([envs.single_action_space.sample() for _ in range(NUM_ENVS)])
    next_states, rewards, terminateds, truncateds, _ = envs.step(actions) 

    for state, action, reward, terminated, truncated, next_state in zip(states, actions, rewards, terminateds, truncateds, next_states):
        if terminated or truncated:
            replay_memory.add(Transition(state, action, reward, True, next_state))
        else:
            replay_memory.add(Transition(state, action, reward, False, next_state))

    states = next_states


# Main Training Loop
states = envs.reset()[0]
for step in range(TOTAL_STEPS):
    epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    beta = np.interp(step, [0, BETA_DECAY], [BETA_START, BETA_END])

    # Get Action
    actions = Q_net.act(states, epsilon)

    # Take Step
    next_states, rewards, terminateds, truncateds, infos = envs.step(actions) 

    # Store Transition and append episode info if needed
    for state, action, reward, terminated, truncated, next_state in zip(states, actions, rewards, terminateds, truncateds, next_states):
        if terminated or truncated:
            replay_memory.add(Transition(state, action, reward, True, next_state))
            episodeinfos_buffer.append(infos['episode'])
            episode_count += 1
        else:
            replay_memory.add(Transition(state, action, reward, False, next_state))

    states = next_states

    # Sample Random Minibatch of Transitions
    minibatch_transitions, importances, indices = replay_memory.sample(BATCH_SIZE, beta)

    minibatch_states = np.asarray([t[0] for t in minibatch_transitions])
    minibatch_states_tensor = torch.as_tensor(minibatch_states, dtype=torch.float32, device=device)

    minibatch_actions = np.asarray([t[1] for t in minibatch_transitions])
    minibatch_actions_tensor = torch.as_tensor(minibatch_actions, dtype=torch.int64, device=device).unsqueeze(-1)

    minibatch_rewards = np.asarray([t[2] for t in minibatch_transitions])
    minibatch_rewards_tensor = torch.as_tensor(minibatch_rewards, dtype=torch.float32, device=device).unsqueeze(-1)

    minibatch_dones = np.asarray([t[3] for t in minibatch_transitions])
    minibatch_dones_tensor = torch.as_tensor(minibatch_dones, dtype=torch.float32, device=device).unsqueeze(-1)

    minibatch_next_states = np.asarray([t[4] for t in minibatch_transitions])
    minibatch_next_states_tensor = torch.as_tensor(minibatch_next_states, dtype=torch.float32, device=device)
    
    importances_tensor = torch.as_tensor(importances, dtype=torch.float32, device=device).unsqueeze(-1)

    # Compute Targets (DDQN Changes Here)
    max_target_online_q_values = Q_net(minibatch_next_states_tensor).argmax(dim=1, keepdim=True)
    target_net_q_values = target_net(minibatch_next_states_tensor)
    target_net_chosen_q_values = torch.gather(input=target_net_q_values, dim=1, index=max_target_online_q_values)
    targets = minibatch_rewards_tensor + GAMMA * (1 - minibatch_dones_tensor) * target_net_chosen_q_values

    # Compute Online Network's Q Values
    q_values = Q_net(minibatch_states_tensor)
    action_q_values = torch.gather(input=q_values, dim=1, index=minibatch_actions_tensor)

    # Update Priorities (Prioritized Experience Replay Changes Here)
    error = targets - action_q_values
    priorities = np.abs(error.detach().cpu().numpy())
    replay_memory.update_priorities(indices, priorities[0])
    
    # Compute Huber Loss (Prioritized Experience Replay Changes Here)
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)
    loss = (loss * importances_tensor).mean()
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update Target Network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(Q_net.state_dict())
    
    # Log Progress
    if step % LOG_INTERVAL == 0:
        rew_mean = np.mean([e['r'] for e in episodeinfos_buffer])
        if math.isnan(rew_mean):
            rew_mean = 0
        len_mean = np.mean([e['l'] for e in episodeinfos_buffer])
        if math.isnan(len_mean):
            len_mean = 0
        # Log to Terminal
        print()
        print(f"Step {step}: AvgEpisodeReward: {rew_mean}, AvgEpisodeLength: {len_mean}, Episodes: {episode_count}")

        # Log to Tensorboard
        summary_writer.add_scalar("AvgEpisodeReward", rew_mean, step)
        summary_writer.add_scalar("AvgEpisodeLength", len_mean, step)
        summary_writer.add_scalar("Episodes", episode_count, step)
    
    # Save Model
    if step % SAVE_INTERVAL == 0 and step > 0:
        print("Saving Model...")
        torch.save(Q_net.state_dict(), SAVE_DIR)

envs.close()