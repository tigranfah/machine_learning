import sys
import os
sys.path.insert(0, os.path.join("gym"))

import gym


env = gym.make("CartPole-v0")
env.reset()
env.render()
