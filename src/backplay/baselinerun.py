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
from baselines.common.vec_env.vec_frame_stack import VecFrameStack


from utils import featurize3d, TrainingAgent
import argparse
import tensorflow as tf
import numpy as np
import json
from gym import spaces


from PPO import *


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
sess = tf.InteractiveSession(config=config)


class Wrapped_Env(Pomme):

    def __init__(self, **kwargs):
        super(Wrapped_Env, self).__init__(**kwargs)

    def step(self, actions):
        all_actions = self.act(self.curr_obs)
        all_actions.insert(self.training_agent, actions[0])

        obs, reward, done, info = super(Wrapped_Env, self).step(all_actions)
        agent_state = featurize3d(obs[self.training_agent], self._step_count, self._max_steps)
        agent_reward = np.array([reward[self.training_agent]])
        done = np.array([done])
        info = np.array([info])
        return agent_state, agent_reward, done, info


    def reset(self):
        self.curr_obs = super(Wrapped_Env,self).reset()
        agent_obs = featurize3d(self.curr_obs[self.training_agent], self._step_count, self._max_steps)
        # agent_obs = featurize3d(obs[self.gym.training_agent], self.gym._step_count, self.gym._max_steps)
        # print(agent_obs.shape)
        return agent_obs


def main():
    # Print all possible environments in the Pommerman registry
    # Instantiate the environment
    DETERMINISTIC = False
    VISUALIZE = False

    if args.test:
        DETERMINISTIC = True
        VISUALIZE = True

    config = ffa_competition_env()
    env = Wrapped_Env(**config["env_kwargs"])
    env.seed(0)
    env.observation_space = spaces.Box(0,20,shape=(11,11,18))
    env.num_envs = 1


    # Add 3 random agents
    agents = []
    for agent_id in range(3):
        agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

    agent_id += 1

    # Add TensorforceAgent
    agents.append(TrainingAgent(config["agent"](agent_id, config["game_type"])))
    env.set_agents(agents)
    env.set_training_agent(agents[-1].agent_id)
    env.set_init_game_state(None)

    # env = VecFrameStack(env, 1)

    # print(env.reset())

    policy = CnnPolicy

    # Model(policy=policy,
    #            ob_space=env.observation_space,
    #            ac_space=env.action_space,
    #            nbatch_act=1,
    #            nbatch_train=100,
    #            nsteps=1000,
    #            ent_coef=0.01,
    #            vf_coef=0.5,
    #            max_grad_norm=0.5)
    num_timesteps=10000

    learn(policy=policy, env=env, nsteps=800, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1))











    #
    # # Instantiate and run the environment for 5 episodes.
    # if VISUALIZE:
    #     wrapped_env = UpdatedEnv(env, True)
    # else:
    #     wrapped_env = UpdatedEnv(env)
    #
    #
    #
    # rewards = []
    # episodes = []
    # def episode_finished(r):
    #     nonlocal episodes
    #     nonlocal rewards
    #     print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
    #                                                                          reward=r.episode_rewards[-1]))
    #     if r.episode % 1000 == 0:
    #         agent.save_model(('./{}').format(EXPERIMENT_NAME), False)
    #         pickle_data(EXPERIMENT_NAME, reward, episodes)
    #     if r.episode_rewards[-1] >= 5:
    #         print()
    #         print()
    #         print()
    #         print("WINNER WINNER CHICKEN DINNER")
    #     episodes.append(r.episode)
    #     rewards.append(r.episode_rewards[-1])
    #     return True
    #
    # # Restore, Train, and Save Model
    # if args.test or args.resume: # If test, change settings and restore model
    #     agent.restore_model('./','PPO_K_someS_500batch_biggerreward_99dis')
    # runner.run(episodes=EPISODES, max_episode_timesteps=800, episode_finished=episode_finished, deterministic=False)
    #
    # if not args.test:
    #     agent.save_model(('./{}').format(EXPERIMENT_NAME), False)
    # print("Stats: ", runner.episode_rewards[-5:], runner.episode_timesteps[-5:])
    #
    # pickle_data(EXPERIMENT_NAME, reward, episodes)
    #
    #
    # try:
    #     runner.close()
    # except AttributeError as e:
    #     pass
    #
    #
    #
    #


if __name__ == '__main__':
    main()
