import numpy as np
import pommerman


"""
Passage = 0
Rigid = 1
Wood = 2
Bomb = 3
Flames = 4
Fog = 5
ExtraBomb = 6
IncrRange = 7
Kick = 8
AgentDummy = 9
Agent0 = 10
Agent1 = 11
Agent2 = 12
Agent3 = 13
"""



def featurize3d(obs, timestep, total_time):
    """
    Create 3D Representation of Pommerman gamestate of a single agent

    Return:
        18x11x11 Numpy Array
    """
    passage = np.where(obs['board'] == 0, 1, 0)
    rigid = np.where(obs['board'] == 1, 1, 0)
    wood = np.where(obs['board'] == 2, 1, 0)
    bomb_pos = np.where(obs['board'] == 3, 1, 0)
    flames = np.where(obs['board'] == 4, 1, 0)
    fog = np.where(obs['board'] == 5, 1, 0)
    pow_bomb = np.where(obs['board'] == 6, 1, 0)
    pow_range = np.where(obs['board'] == 7, 1, 0)
    pow_kick = np.where(obs['board'] == 8, 1, 0)

    bomb_blast_strength = obs['bomb_blast_strength']
    bomb_life = obs['bomb_life']

    if obs['can_kick']:
        can_kick = np.ones_like(obs['board'])
    else:
        can_kick = np.zeros_like(obs['board'])

    bomb_count = np.ones_like(obs['board']) * obs['ammo']
    agent_blast_strength = np.ones_like(obs['board']) * obs['blast_strength']


    agent = np.zeros_like(obs['board'])
    agent_x, agent_y = obs['position']
    agent[agent_x, agent_y] = 1

    enemy1 = np.where(obs['board'] == obs['enemies'][0].value, 1, 0)
    enemy2 = np.where(obs['board'] == obs['enemies'][2].value, 1, 0)

    wild_card, has_teammate = determine_teammate(obs)

    time_left = np.ones_like(obs['board']) * (timestep / total_time)

    # 18x11x11 Observation
    observation = np.stack([bomb_blast_strength, bomb_life, agent, bomb_count, agent_blast_strength,
                            can_kick, has_teammate, wild_card, enemy1, enemy2, passage, rigid, wood,
                            flames, pow_bomb, pow_range, pow_kick, time_left])

    return observation

def determine_teammate(obs):

    if obs['teammate'].value == pommerman.constants.Item.AgentDummy:
        wild_card = np.where(obs['board'] == obs['enemies'][1].value, 1, 0)
        has_teammate = np.zeros_like(obs['board'])
    else:
        wild_card = np.where(obs['board'] == obs['teammate'].value, 1, 0)
        has_teammate = np.ones_like(obs['board'])
    return wild_card, has_teammate




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

        # agent_state = featurize(obs[self.gym.training_agent])
        agent_state = featurize3d(state[self.gym.training_agent], self.gym._step_count, self.gym._max_steps)
        agent_reward = reward[self.gym.training_agent]

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
        # agent_obs = featurize(obs[self.gym.training_agent])
        agent_obs = featurize3d(obs[self.gym.training_agent], self.gym._step_count, self.gym._max_steps)
        print(agent_obs.shape)
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
