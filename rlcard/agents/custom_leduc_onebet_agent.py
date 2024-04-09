import numpy as np


class CustomLeducOneBetAgent(object):
    ''' Our custom Leduc One Bet agent with probabilities of actions
        derived from our LP analytical solution
    '''

    def __init__(self, num_actions=2):
        ''' Initialize the custom agent
        
        Args:
            num_actions (int): The size of the output action space
        '''
        self.use_raw = True
        self.num_actions = num_actions

        # defining the probabilities of actions
        # p1 first action 
        self.p1_raise_check_probs = {
            # (public_rank, self_rank): probability of first action
            ('J', 'J'): 1.00,
            ('J', 'Q'): 0.25,
            ('J', 'K'): 0.25,
            ('Q', 'J'): 0.25,
            ('Q', 'Q'): 1.00,
            ('Q', 'K'): 0.25,
            ('K', 'J'): 0.00,
            ('K', 'Q'): 0.00,
            ('K', 'K'): 0.00
        }

        # given p2 raises
        self.p1_call_fold_probs = {
            # (public_rank, self_rank): probability of second action
            ('J', 'J'): None,
            ('J', 'Q'): 0.00,
            ('J', 'K'): 1.00,
            ('Q', 'J'): None,
            ('Q', 'Q'): 0.00,
            ('Q', 'K'): 1.00,
            ('K', 'J'): 0.00,
            ('K', 'Q'): 0.00,
            ('K', 'K'): 1.00
        }

        # given p1 checks
        self.p2_raise_check_probs = {
            ('J', 'J'): 1.00,
            ('J', 'Q'): 0.25,
            ('J', 'K'): 0.25,
            ('Q', 'J'): 0.25,
            ('Q', 'Q'): 1.00,
            ('Q', 'K'): 0.25,
            ('K', 'J'): 2/3,
            ('K', 'Q'): 2/3,
            ('K', 'K'): 0.00
        }

        # given p1 raises
        self.p2_call_fold_probs = {
            ('J', 'J'): 1.00,
            ('J', 'Q'): 0.00,
            ('J', 'K'): 0.75,
            ('Q', 'J'): 0.00,
            ('Q', 'Q'): 1.00,
            ('Q', 'K'): 0.75,
            ('K', 'J'): 2/3,
            ('K', 'Q'): 2/3,
            ('K', 'K'): 1.00
        }
        

    def step(self, state):
        ''' Predict the action given the current state
        
        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by the custom agent
        '''
        action = self.get_action(state['raw_obs'], state['action_record'])
        return action
    
    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation
        
        Args:
            state (numpy.array): An array that represents the current state

        Returns:
            action (int): The action predicted by the custom agent
        '''
        return self.step(state), {}
    
    def get_action(self, raw_obs, action_record):
        ''' Obtain the action at the current state. 
            Uses the predefined probabilities of actions to decide.
        
        Args:
            raw_obs (dict): Dict representation of the current state
            legal_actions (list): List of legal actions

        Returns:
            action (str): The action decided by the custom agent
        '''
        public_rank = raw_obs['public_card'][-1]
        self_rank = raw_obs['hand'][-1]

        # If record is empty, means we are in the first round
        # and we play first
        if not action_record:
            prob = self.p1_raise_check_probs[(public_rank, self_rank)]
            return np.random.choice(['raise', 'check'], p=[prob, 1-prob])
        
        # we are player 2 
        elif len(action_record) == 1:
            if action_record[0][1] == 'check':
                prob = self.p2_raise_check_probs[(public_rank, self_rank)]
                return np.random.choice(['raise', 'check'], p=[prob, 1-prob])
            elif action_record[0][1]  == 'raise':
                prob = self.p2_call_fold_probs[(public_rank, self_rank)]
                return np.random.choice(['call', 'fold'], p=[prob, 1-prob])
            else:
                raise('P1 folds?')

        # we are player 1
        # player 2 raises, we call or fold
        elif len(action_record) == 2:
            prob = self.p1_call_fold_probs[(public_rank, self_rank)]
            return np.random.choice(['call', 'fold'], p=[prob, 1-prob])
        else: raise('Invalid action record length')
