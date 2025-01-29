import gymnasium as gym
from collections import deque
import itertools
import numpy as np
import math
import ale_py
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

gym.register_envs(ale_py)
env = gym.make("ALE/SpaceInvaders-v5")
env = EpisodicLifeEnv(env)
env = NoopResetEnv(env, noop_max=30)
env = MaxAndSkipEnv(env, skip=4)
env = FireResetEnv(env)
env = Monitor(env)

episodeinfos_buffer = deque([], maxlen=100)
episode_count = 0

summary_writer = SummaryWriter("./logs/random_agent")

obs, info = env.reset()
for step in range(int(5e6)):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        episode_count += 1
        episodeinfos_buffer.append(info['episode'])
        obs, info = env.reset()

    if step % 1000 == 0:
        rew_mean = np.mean([e['r'] for e in episodeinfos_buffer])
        if math.isnan(rew_mean):
            rew_mean = 0
        len_mean = np.mean([e['l'] for e in episodeinfos_buffer])
        if math.isnan(len_mean):
            len_mean = 0
        # Log to Terminal 
        print()
        print(f"Steps: {step} Episode: {episode_count} Reward: {rew_mean:.2f} Length: {len_mean:.2f}")

        # Log to Tensorboard
        summary_writer.add_scalar("AvgEpisodeReward", rew_mean, step)
        summary_writer.add_scalar("AvgEpisodeLength", len_mean, step)
        summary_writer.add_scalar("Episodes", episode_count, step)

env.close()