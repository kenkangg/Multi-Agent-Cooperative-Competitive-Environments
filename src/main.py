import os
import sys
import numpy as np

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v1_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility


config = ffa_v1_env()
env = Pomme(**config["env_kwargs"])
env.seed(0)
