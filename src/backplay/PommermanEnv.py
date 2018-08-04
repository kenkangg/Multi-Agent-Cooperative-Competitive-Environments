#!/usr/bin/env python3
import sys
from baselines import logger
# from baselines.common.cmd_util import atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
import ppo2
# from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
import multiprocessing
import tensorflow as tf
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env, ffa_competition_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
import pickle
from gym import spaces
from utils import featurize3d, TrainingAgent
import numpy as np
from baselines import logger
from baselines.bench import Monitor
import os

from PPO import CnnPolicy

class Wrapped_Env(Pomme):

    def __init__(self, **kwargs):
        super(Wrapped_Env, self).__init__(**kwargs)
        #Dictionary of Demonstration, Winner ID
        # data = pickle.load(open( "demonstration.p", "rb" ))
        # self.demo = data['demo']
        # self.winner_id = data['winner']
        #
        # self.windows = [[4,16],[10,40],[30,90],[80,240],[200,500],[400,800],[800,800]]
        # self.ep = [0,50,100,150,200,250,300]


    ##ADDED
    # def get_initial_state(self, step):
    #     for i in range(len(self.ep) - 1):
    #         if self.ep[i+1] >= step:
    #             windows = self.windows[i]
    #             break
    #     if step > self.ep[-1]:
    #         windows = [800,800]
    #     print(windows)
    #
    #     reverse = self.demo[::-1]
    #     return random.choice(reverse[windows[0]:windows[1]])


    def step(self, actions):
        all_actions = self.act(self.curr_obs)
        all_actions.insert(self.training_agent, actions)
        # print(all_actions)

        obs, reward, done, info = super(Wrapped_Env, self).step(all_actions)
        agent_state = featurize3d(obs[self.training_agent], self._step_count, self._max_steps)
        agent_reward = reward[self.training_agent]

        if agent_reward == 1 or agent_reward == -1:
            episode = {'r': agent_reward, 'l': self._step_count}
            info['episode'] = episode

        return agent_state, agent_reward, done, info

    # def get_initial_gamestate(self, update_step):
    #     initial_state = self.get_initial_state(update_step)
    #
    #     self._init_game_state = initial_state
    #     self.set_json_info()
    #     self.training_agent = self.winner_id
    #
    #
    #     self.curr_obs = self.get_observations()
    #     agent_obs = featurize3d(self.curr_obs[self.training_agent], self._step_count, self._max_steps)
    #     return agent_obs

    def reset(self):
        self.curr_obs = super(Wrapped_Env,self).reset()
        agent_obs = featurize3d(self.curr_obs[self.training_agent], self._step_count, self._max_steps)
        # agent_obs = featurize3d(obs[self.gym.training_agent], self.gym._step_count, self.gym._max_steps)
        # print(agent_obs.shape)
        return agent_obs



def train():
    logger.configure()

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()


    ##### POMMERMAN

    def make_env(seed):
        def f():
            config = ffa_competition_env()
            env = Wrapped_Env(**config["env_kwargs"])
            env.observation_space = spaces.Box(0,20,shape=(11,11,18), dtype=np.float32)

            # Add 3 random agents
            agents = []
            for agent_id in range(3):
                # if agent_id == env.winner_id:
                #     agents.append(TrainingAgent(config["agent"](agent_id, config["game_type"])))
                # else:
                agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))
            agent_id += 1
            agents.append(TrainingAgent(config["agent"](agent_id, config["game_type"])))

            env.set_agents(agents)
            env.set_training_agent(agents[-1].agent_id)
            env.set_init_game_state(None)

            if logger.get_dir():
                env = Monitor(env, logger.get_dir(), allow_early_resets=True)

            return env
        return f

    #########
    envs = [make_env(seed) for seed in range(8)]
    env = SubprocVecEnv(envs)

    num_timesteps = 10000
    policy = CnnPolicy
    # env = VecFrameStack(make_atari_env(env_id, 8, seed), 4)
    # policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy, 'mlp': MlpPolicy}[policy]
    model= ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1))


    logger.log("Running trained model")
    # obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    env = make_env(0)()
    obs = env.reset()
    obs = np.expand_dims(obs,0)
    while True:
        print(obs.shape)
        actions = model.step(obs)[0]
        obs[:], reward, done, info  = env.step(actions)

        if done:
            obs = env.reset()
            obs = np.expand_dims(obs,0)

        env.render()



if __name__ == '__main__':
    train()
