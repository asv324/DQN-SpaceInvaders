from torch import nn
import torch
import gymnasium as gym
import itertools
import ale_py
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)
from stable_baselines3.common.monitor import Monitor

# change this path to the model you want to load
LOAD_DIR = "./models/vanilla_dqn"

gym.register_envs(ale_py)
def make_env():
    env = gym.make("ALE/SpaceInvaders-v5", repeat_action_probability=0.0, render_mode="human")
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = FireResetEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    return env

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.num_actions = env.single_action_space.n
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

    def forward(self, x):
        return self.net(x / 255.0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.vector.SyncVectorEnv([lambda: Monitor(make_env()) for _ in range(1)])

net = Network(env).to(device)
net.load_state_dict(torch.load(LOAD_DIR, map_location=device))

states = env.reset()[0]
for step in itertools.count():
    q_values = net(torch.Tensor(states).to(device))
    actions = torch.argmax(q_values, dim=1).cpu().numpy()


    states, rewards, terminated, truncated, infos = env.step(actions)

    if terminated or truncated:
        states = env.reset()[0]
