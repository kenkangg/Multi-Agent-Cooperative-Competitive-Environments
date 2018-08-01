import pommerman
from pommerman.agents import SimpleAgent, PlayerAgent
from utils import *
from pommerman.configs import ffa_v1_env
from pommerman.envs.v0 import Pomme
import pickle

import sys
from baselines import logger
# from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
import multiprocessing
import tensorflow as tf


num_timesteps = 1000


def main():
    # Print all possible environments in the Pommerman registry
    print(pommerman.registry)

    config = ffa_v1_env()
    env = Pomme(**config["env_kwargs"])
    env.num_envs = 1

    # Add 3 agents
    agents = {}
    for agent_id in range(4):
        agents[agent_id] = SimpleAgent(config["agent"](agent_id, config["game_type"]))


    env.set_agents(list(agents.values()))
    env.set_init_game_state(None)

    # Run the episodes just like OpenAI Gym
    policy = CnnPolicy
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1))


if __name__ == '__main__':
    main()
