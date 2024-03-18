import json
import os

import rlcard
from rlcard.envs.leducholdem import LeducholdemEnv as Env
from rlcard.games.leduc_onebet import Game


class LeducOneBetEnv(Env):
    def __init__(self, config):
        ''' Initialize the class leducholdem OneBet Env
        '''
        self.game = Game()
        self.name = 'leduc-onebet'
        super(Env, self).__init__(config)       # inherit from rlcard.envs.Env instead
        self.actions = ['call', 'raise', 'fold', 'check']
        self.state_shape = [[36] for _ in range(self.num_players)]  # TODO: we dont need this much space
        self.action_shape = [None for _ in range(self.num_players)]

        with open(os.path.join(rlcard.__path__[0], 'games/leducholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)