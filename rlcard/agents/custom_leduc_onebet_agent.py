import numpy as np


class CustomLeducOneBetAgent(object):
    ''' Our custom Leduc One Bet agent with probabilities of actions
        derived from our LP analytical solution
    '''

    def __init__(self, num_actions):
        ''' Initialize the custom agent
        
        Args:
            num_actions (int): The size of the output action space
        '''
        self.use_raw = False
        self.num_actions = num_actions

    def step(self, state):
        ''' Predict the action given the current state
        
        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by the custom agent
        '''
        legal_actions = list(state['legal_actions'].keys())
        action_probs = self.action_probs(state['obs'], legal_actions)
        action = np.random.choice(len(action_probs), p=action_probs)
        return legal_actions[action]
    
    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation
        
        Args:
            state (numpy.array): An array that represents the current state

        Returns:
            action (int): The action predicted by the custom agent
        '''
        return self.step(state), {}
    
    def action_probs(self, obs, legal_actions):
        ''' Obtain the action probabilities of the current state
        
        Args:
            obs (str?): State string
            legal_actions (list): List of legal actions

        Returns:
            action_probs (numpy.array): The action probabilities
        '''
        action_probs = np.array([1/len(legal_actions) for _ in range(legal_actions)])
        return action_probs
