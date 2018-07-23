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
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.agents import Agent

from pommerman import utility

import argparse

import tensorflow as tf

import numpy as np

import json


parser = argparse.ArgumentParser()
parser.add_argument('-test', action='store_true')
parser.add_argument('-resume', action='store_true')
parser.add_argument('-exp', type=str)
args = parser.parse_args()

print(args.test)

EXPERIMENT_NAME = args.exp
BATCH_SIZE = 32
VISUALIZE = False
EPISODES = 100000
# EXPERIMENT_NAME = 'model_timestep_reward'

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)



def make_np_float(feature):
    return np.array(feature).astype(np.float32)

def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))

class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass

class WrappedEnv(OpenAIGym):
    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize

    def execute(self, actions):
        if self.visualize:
            self.gym.render()

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)


        alive_before = self.check_alive()
        bomb_before = self.check_bombs()
        state, reward, terminal, _ = self.gym.step(all_actions)

        alive_after = self.check_alive()
        bomb_after = self.check_bombs()

        agent_state = featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        ## print(bomb_before, bomb_after)
        if agent_reward == 1:
            agent_reward *= 5
        elif alive_before != alive_after and self.gym._agents[self.gym.training_agent].is_alive and alive_after != 1:
            print(alive_before, alive_after)
            # agent_reward += 0.25 * (alive_before - alive_after)
            if bomb_before < bomb_after:
                print(bomb_before, bomb_after)
                agent_reward += 1  * (alive_before - alive_after)
        # agent_reward -= 0.001
        return agent_state, terminal, agent_reward

    def check_alive(self):
        """ Self Added """
        count = 0
        for agent in self.gym._agents:
            if agent.is_alive:
                count += 1
        return count

    def check_bombs(self):
        """ Self Added """
        obs = self.gym.get_observations()
        return utility.make_np_float([obs[0]["ammo"]])

    def reset(self):
        obs = self.gym.reset()
        agent_obs = featurize(obs[3])
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
    env = Pomme(**config["env_kwargs"])
    env.seed(0)

    # Create a Proximal Policy Optimization agent
    with open('ppo.json', 'r') as fp:
            agent = json.load(fp=fp)

    with open('mlp2_lstm_network.json', 'r') as fp:
            network = json.load(fp=fp)

    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states=dict(type='float', shape=env.observation_space.shape),
            actions=dict(type='int', num_actions=env.action_space.n),
            network=network
        )
    )

    # Add 3 random agents
    agents = []
    for agent_id in range(3):
        agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

    # Add TensorforceAgent
    agent_id += 1
    agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))
    env.set_agents(agents)
    env.set_training_agent(agents[-1].agent_id)
    env.set_init_game_state(None)

    # Instantiate and run the environment for 5 episodes.
    if VISUALIZE:
        wrapped_env = WrappedEnv(env, True)
    else:
        wrapped_env = WrappedEnv(env)

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
            try:
                prev_data = pickle.load(open(EXPERIMENT_NAME, "rb"))
                prev_len = len(prev_data[0])
                prev_data[0].extend(rewards)
                rewards = []
                prev_data[1].extend(episodes)
                episodes = []
                pickle.dump(prev_data, open(EXPERIMENT_NAME, "wb"))
            except (OSError, IOError) as e:
                pickle.dump([rewards, episodes], open(EXPERIMENT_NAME, "wb"))
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
    runner.run(episodes=EPISODES, max_episode_timesteps=2000, episode_finished=episode_finished, deterministic=False)

    if not args.test:
        agent.save_model(('./{}').format(EXPERIMENT_NAME), False)
    print("Stats: ", runner.episode_rewards[-5:], runner.episode_timesteps[-5:])

    #Dump reward values
    try:
        prev_data = pickle.load(open(EXPERIMENT_NAME, "rb"))
        prev_len = len(prev_data[0])
        prev_data[0].extend(rewards)
        prev_data[1].extend(episodes)
        print(episodes)
        pickle.dump(prev_data, open(EXPERIMENT_NAME, "wb"))
    except (OSError, IOError) as e:
        pickle.dump([rewards, episodes], open(EXPERIMENT_NAME, "wb"))

    try:
        runner.close()
    except AttributeError as e:
        pass



if __name__ == '__main__':
    main()
