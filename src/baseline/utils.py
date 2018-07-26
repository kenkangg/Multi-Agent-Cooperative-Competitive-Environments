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
import numpy as np
import pickle




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

class UpdatedEnv(OpenAIGym):
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
            #agent_reward += 0.25 * (alive_before - alive_after)
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

    def pickle_data(name, reward, episodes):
        try:
            prev_data = pickle.load(open(EXPERIMENT_NAME, "rb"))
            prev_len = len(prev_data[0])
            prev_data[0].extend(rewards)
            prev_data[1].extend(episodes)
            print(episodes)
            pickle.dump(prev_data, open(EXPERIMENT_NAME, "wb"))
        except (OSError, IOError) as e:
            pickle.dump([rewards, episodes], open(EXPERIMENT_NAME, "wb"))
