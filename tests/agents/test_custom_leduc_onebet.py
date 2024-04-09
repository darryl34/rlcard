import unittest

import rlcard
from rlcard.agents.custom_leduc_onebet_agent import CustomLeducOneBetAgent
from rlcard.agents.random_agent import RandomAgent

class TestCustomLeducOneBet(unittest.TestCase):

    def test_init(self):
        agent = CustomLeducOneBetAgent()
        self.assertEqual(agent.num_actions, 2)


    def test_env(self):
        env = rlcard.make('leduc-onebet')
        custom_agent = CustomLeducOneBetAgent()
        random_agent = RandomAgent(num_actions=env.num_actions)
        env.set_agents([custom_agent, random_agent])

