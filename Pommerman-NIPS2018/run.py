import pommerman
from pommerman import agents
from agent import NothingAgent

import numpy as np

"""
State Space: Alive [<=4]
             Board {0: open space,
                    1: unbreakable,
                    2: breakable}
             Bomb Blast strength
             Bomb_life
             Position
             Blast_strength
             Can_Kick,
             Teammate
             Ammo
             Enemies

"""



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

def main():
    # Print all possible environments in the Pommerman registry
    print(pommerman.registry)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=5000),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFAFast-v0', agent_list)

    agent = NothingAgent(env)
    agent_list.append(agent)
    env.set_agents(agent_list)


    env.seed(0)
#
    # Run the episodes just like OpenAI Gym
    for i_episode in range(10):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
