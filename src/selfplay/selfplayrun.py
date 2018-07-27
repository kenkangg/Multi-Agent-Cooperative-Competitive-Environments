### TODO: Make all agents same
# TODO: SelfPlay: Collect data from self PlayerAgent
# TODO: Optimizer: Train on game data, every N iterations test against old agent
# TODO: Evaluator: If new agent is better than old agent, old agent == new agents
# TODO: ^^^Asyncronous

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env, ffa_competition_env, ffa_v1_env
from pommerman.envs.v0 import Pomme
from utils import featurize, WrappedEnv, learn

from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
import tensorflow as tf
import numpy as np

EPISODES = 200
MAX_EPISODE_LENGTH = 1000
VISUALIZE=False



config = tf.ConfigProto()
tf.Session(config=config).__enter__()


# Instantiate the environment
config = ffa_v1_env()
env = Pomme(**config["env_kwargs"])
env.seed(0)


# Add 3 random agents
agents = []
for agent_id in range(3):
    agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

# Add TensorforceAgent
agent_id += 1
agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))
env.set_agents(agents)
env.set_training_agent(agents[-1].agent_id)
env.set_init_game_state(None)
env = WrappedEnv(env, visualize=VISUALIZE)


###
### Create PPO Model
###
nsteps=128
vf_coef=0.5
nminibatches=4
max_grad_norm=0.5
nenvs = 1
nbatch = nenvs * nsteps
nbatch_train = nbatch // nminibatches


policy_network = MlpPolicy

# model = ppo2.Model(policy=policy_network,
#                        ob_space=env.gym.observation_space,
#                        ac_space=env.gym.action_space,
#                        nbatch_act=nenvs,
#                        nbatch_train=nbatch_train,
#                        nsteps=nsteps,
#                        ent_coef=0.01,
#                        vf_coef=vf_coef,
#                        max_grad_norm=max_grad_norm)

num_timesteps = 1000000

learn(policy=policy_network, env=env, nsteps=128, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1))

# policy = model.act_model
# # Run Episodes
# for i_episode in range(EPISODES):
#     observation = env.reset()
#     for t in range(MAX_EPISODE_LENGTH):
#         actions = [policy.step(obs) for obs in observation]
#         actions = [action[0] for action in actions]
#         # print(actions)
#         observation, reward, done = env.step(actions)
#         if done[env.gym.training_agent]:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
