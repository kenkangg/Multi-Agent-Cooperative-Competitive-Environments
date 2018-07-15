
from pommerman import agents
from pommerman import characters

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import numpy as np

class NothingAgent(agents.BaseAgent):
    def initialize_agent(self, env):
        np.random.seed(123)
        env.seed(123)
        nb_actions = 6

        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        print(model.summary())

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                       target_model_update=1e-2, policy=policy)
        agent.compile(Adam(lr=1e-3), metrics=['mae'])

        self.agent = agent



    def act(self, obs, action_space):
        """
        0: Stay
        1: Right?
        2: Down
        3: Left
        4: Up?
        5: Place Bomb
        """
        policy = self.agent.forward(featurize(obs))
        # print(policy)
        return policy




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
