import gym
from PPO import PPOAgent
import tensorflow as tf
import numpy as np

"""
- Collect data
- Train on that data every n episodes
"""

sess = tf.InteractiveSession()
env = gym.make('CartPole-v0')


agent = PPOAgent(env)





for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = agent.network.act(observation)[0]

        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
