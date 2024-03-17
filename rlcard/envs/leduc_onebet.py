from rlcard.envs.leducholdem import Env
from rlcard.games.leduc_onebet import Game


class LeducOneBetEnv(Env):
    def __init__(self, config):
        ''' Initialize the class leducholdem OneBet Env
        '''
        self.game = Game()
        self.name = 'leduc-onebet'
        super().__init__(config)
