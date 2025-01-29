# Playing Space Invaders using DQN
## A complete implementation of DQN, Prioritized Experience Replay and Double DQN
In this project, I trained an agent using Deep Reinforcement Learning to play the iconic Atari 2600 game of Space Invaders. In particular, I employed the [DQN algorithm](https://www.nature.com/articles/nature14236) to train the agent and also added two improvements:

* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
* [Double DQN](https://arxiv.org/abs/1509.06461)

## Watch how the trained DQN agent plays compared to a random agent!
https://github.com/user-attachments/assets/b11483e8-b369-4105-b049-32d64c3a0b4d

## Space Invaders as an environment
For this project, I have used the [Arcade Learning Environment](https://ale.farama.org/index.html) to emulate the game and the [Gymnasium API](https://gymnasium.farama.org/index.html) to allow our agents to interact with the game. The game screen has a 210x160x3 RGB image, resulting in 100,800 colour values per frame. Relevant game elements include the player’s cannon, multiple rows of aliens (each with different point values), protective barriers, and occasionally a command alien ship.

### Environment Preprocessing
I stuck to the same preprocessing steps used [in the original DQN paper](https://www.nature.com/articles/nature14236). I used certain Atari wrappers from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html) to achieve this:

* Used the MaxAndSkipEnv() function to return only every k-th frame (frameskipping) and return the max between the two last frames.
* Used the NoopResetEnv() function to sample initial states by taking random number of no-ops on reset.
* Used the EpisodicLifeEnv() function to make end-of-life == end-of-episode, but only reset on true game over.
* Used the FireResetEnv() function to take action on reset for environments that are fixed until firing.

I also used the following wrappers from Gymnasium:

* Used the ResizeObservation() to reshape the image to 84 by 84 pixels.
* Used the GrayscaleObservation() to convert image to grayscale.
* Used the FrameStackObservation() to stack the four most recent frames to use as the agent's observation.

## Network Architecture and Training Details
The network used follows the same structure as the network used [in the original DQN paper](https://www.nature.com/articles/nature14236). The neural network takes an 84x84x4 image as input. The first hidden layer applies 32 convolutional filters of size 8x8 with a stride of 4, followed by a rectifier activation function. The second hidden layer uses 64 convolutional filters of size 4x4 with a stride of 2, also followed by a rectifier activation. Next, a third convolutional layer employs 64 filters of size 3x3 with a stride of 1, again followed by a rectifier activation. The final hidden layer is fully connected, consisting of 512 rectifier units. The output layer is a fully connected linear layer that provides one output per valid action (6).

I trained two agents: vanilla DQN and Double DQN with prioritized experience replay. I trained all models for 5 million steps. For hyperparameter selection, I simply used the optimal hyperparameter combinations specified by [in the original DQN paper](https://www.nature.com/articles/nature14236) for the vanilla DQN model. For the combined Double DQN with prioritized replay model, I used the [chosen hyperparameters for proportional prioritized experience replay](https://arxiv.org/abs/1511.05952).

## Learning Curves comparing our trained agents
![Average Reward of Different Agents](https://github.com/user-attachments/assets/cc0afd66-d169-4d31-b0a2-7ec2cf8acdb7)
![Average Episode Length of Different Agents](https://github.com/user-attachments/assets/e078c917-4090-43ec-bff0-6289c75c7927)

## How to train/load agents using this repository
First, run this command to install the dependencies
    
    pip install -r requirements.txt

*You may also need this for some reason*
    
    pip install "gymnasium[other]"

  Now to train a model, simply run either "DQN.py" to train a DQN agent or "PER+DDQN.py" to train a PER+DDQN agent. Make sure the logs in your "logs" folder corresponding to your respective agent has been cleared before training, as otherwise the tensorboard graph will look confusing.

  To load a trained model and see it play the game, open "test_agent_playing.py" and change the "LOAD_DIR" constant to the path to your saved trained model in your "models" folder. Now simply run the file to see your agent play! (P.S there are already pre-trained agents in the models folder for you to load and see how they play!)
