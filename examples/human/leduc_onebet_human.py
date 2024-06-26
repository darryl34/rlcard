''' A toy example of playing against pretrained AI on Leduc Hold'em
'''

import os
import sys

sys.path.insert(0, os.getcwd())

import rlcard
from rlcard import models
from rlcard.agents import LeducholdemHumanAgent as HumanAgent
from rlcard.agents import CustomLeducOneBetAgent as CustomAgent
from rlcard.utils import print_card

# Make environment
env = rlcard.make('leduc-onebet', config={'seed': 0})
human_agent = HumanAgent(env.num_actions)
# cfr_agent = models.load('leduc-holdem-cfr').agents[0]
custom_agent = CustomAgent()
env.set_agents([custom_agent, human_agent])

print(">> Leduc Hold'em pre-trained model")

while True:
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record) + 1):
        if action_record[-i][0] == state['current_player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('===============     CFR Agent    ===============')
    print_card(env.get_perfect_information()['hand_cards'][1])

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    input("Press any key to continue...")