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
LR = 2.5e-4
LOG_DIR = "./logs/DQN" + str(LR)
LOG_INTERVAL = 1000
SAVE_DIR = "./models/DQN"
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


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

replay_memory = ReplayMemory(TOTAL_REPLAY_SIZE)
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
            replay_memory.push((state, action, reward, True, next_state))
        else:
            replay_memory.push((state, action, reward, False, next_state))

    states = next_states

# Main Training Loop
states = envs.reset()[0]
for step in range(TOTAL_STEPS):
    epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    # Get Action
    actions = Q_net.act(states, epsilon)

    # Take Step
    next_states, rewards, terminateds, truncateds, infos = envs.step(actions) 

    # Store Transition and append episode info if needed
    for state, action, reward, terminated, truncated, next_state in zip(states, actions, rewards, terminateds, truncateds, next_states):
        if terminated or truncated:
            replay_memory.push((state, action, reward, True, next_state))
            episodeinfos_buffer.append(infos['episode'])
            episode_count += 1
        else:
            replay_memory.push((state, action, reward, False, next_state))

    states = next_states

    # Sample Random Minibatch of Transitions
    minibatch_transitions = replay_memory.sample(BATCH_SIZE)

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
    
    # Compute Targets
    max_target_q_values = target_net(minibatch_next_states_tensor).max(dim=1, keepdim=True)[0]
    targets = minibatch_rewards_tensor + GAMMA * (1 - minibatch_dones_tensor) * max_target_q_values

    # Compute Online Network's Q Values
    q_values = Q_net(minibatch_states_tensor)
    action_q_values = torch.gather(input=q_values, dim=1, index=minibatch_actions_tensor)

    # Compute Huber Loss
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)
    
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