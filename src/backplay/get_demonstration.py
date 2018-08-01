import pommerman
from pommerman.agents import SimpleAgent, PlayerAgent
from utils import *
from pommerman.configs import ffa_v1_env
from pommerman.envs.v0 import Pomme
import pickle


def main():
    # Print all possible environments in the Pommerman registry
    print(pommerman.registry)

    config = ffa_v1_env()
    env = Pomme(**config["env_kwargs"])

    # Add 3 agents
    agents = {}
    for agent_id in range(4):
        agents[agent_id] = SimpleAgent(config["agent"](agent_id, config["game_type"]))

    # agents[3] = PlayerAgent(config["agent"](agent_id, config["game_type"]), "arrows")

    env.set_agents(list(agents.values()))
    env.set_init_game_state(None)

    demo = []

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        demo.append(env.get_json_info())
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            demo.append(env.get_json_info())
        if 1 in reward:
            winner = reward.index(1)
        else:
            winner = None

        print('Episode {} finished'.format(i_episode))
    env.close()

    # If game not tied, save demonstration
    if winner is not None:
        demonstration = {'demo': demo, 'winner': winner}
        pickle.dump( demonstration, open( "demonstration.p", "wb" ) )


if __name__ == '__main__':
    main()
