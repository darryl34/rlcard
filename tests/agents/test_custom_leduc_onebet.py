import unittest

import rlcard
from rlcard.agents.custom_leduc_onebet_agent import CustomLeducOneBetAgent
from rlcard.agents.random_agent import RandomAgent

class TestCustomLeducOneBet(unittest.TestCase):

    def test_init(self):
        agent = CustomLeducOneBetAgent(2)
        self.assertEqual(agent.num_actions, 2)

    