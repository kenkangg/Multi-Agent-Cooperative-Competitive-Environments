import pommerman
import gym

import os
import sys
import numpy as np
import pickle

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env, ffa_competition_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility

from tensorforce.agents import PPOAgent
from tensorforce.agents import DQNAgent
from Runner import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.agents import Agent

from utils import *
import argparse
import tensorflow as tf
import numpy as np
import json
from gym import spaces


parser = argparse.ArgumentParser()
parser.add_argument('-test', action='store_true')
parser.add_argument('-resume', action='store_true')
parser.add_argument('-exp', type=str)
args = parser.parse_args()

EXPERIMENT_NAME = args.exp
BATCH_SIZE = 32
VISUALIZE = False
EPISODES = 50000
# EXPERIMENT_NAME = 'model_timestep_reward'

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


def main():
    # Print all possible environments in the Pommerman registry
    # Instantiate the environment
    DETERMINISTIC = False
    VISUALIZE = False

    if args.test:
        DETERMINISTIC = True
        VISUALIZE = True

    config = ffa_competition_env()
    env = Pomme(**config["env_kwargs"])
    env.seed(0)

    # Create a Proximal Policy Optimization agent
    with open('ppo.json', 'r') as fp:
            agent = json.load(fp=fp)

    with open('mlp2_lstm_network.json', 'r') as fp:
            network = json.load(fp=fp)

    env.observation_space = spaces.Box(0,10,shape=(18,11,11))

    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states=dict(type='float', shape=(env.observation_space.shape)),
            actions=dict(type='int', num_actions=env.action_space.n),
            network=network
        )
    )


    print('here')
    # Add 3 random agents
    agents = []
    for agent_id in range(3):
        agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

    # Add TensorforceAgent
    # agent_id += 1
    agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))
    env.set_agents(agents)
    env.set_training_agent(agents[-1].agent_id)
    env.set_init_game_state(None)

    # Instantiate and run the environment for 5 episodes.
    if VISUALIZE:
        wrapped_env = UpdatedEnv(env, True)
    else:
        wrapped_env = UpdatedEnv(env)

    runner = Runner(agent=agent, environment=wrapped_env)

    rewards = []
    episodes = []
    def episode_finished(r):
        nonlocal episodes
        nonlocal rewards
        print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                             reward=r.episode_rewards[-1]))
        if r.episode % 1000 == 0:
            agent.save_model(('./{}').format(EXPERIMENT_NAME), False)
            pickle_data(EXPERIMENT_NAME, reward, episodes)
        if r.episode_rewards[-1] >= 5:
            print()
            print()
            print()
            print("WINNER WINNER CHICKEN DINNER")
        episodes.append(r.episode)
        rewards.append(r.episode_rewards[-1])
        return True

    # Restore, Train, and Save Model
    if args.test or args.resume: # If test, change settings and restore model
        agent.restore_model('./','PPO_K_someS_500batch_biggerreward_99dis')
    runner.run(episodes=EPISODES, max_episode_timesteps=800, episode_finished=episode_finished, deterministic=False)

    if not args.test:
        agent.save_model(('./{}').format(EXPERIMENT_NAME), False)
    print("Stats: ", runner.episode_rewards[-5:], runner.episode_timesteps[-5:])

    pickle_data(EXPERIMENT_NAME, reward, episodes)


    try:
        runner.close()
    except AttributeError as e:
        pass






if __name__ == '__main__':
    main()
