''' An example of evluating the trained models in RLCard
'''
import os
import sys
import argparse

sys.path.insert(0, os.getcwd())
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import rlcard
from rlcard.agents import (
    DQNAgent,
    RandomAgent,
    CustomLeducOneBetAgent as CustomAgent
)
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)



def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
    
    return agent


# pairwise comparison between 
# (1) among DQN, CFR and custom LP agent
# (2) DQN, CFR and custom with random agent (baseline)
def evaluate(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make('leduc-onebet')

    # Load models
    dqn_agent = load_model('experiments/leduc_holdem_dqn_result/model.pth', env, 0, device)
    cfr_agent = load_model('experiments/leduc_holdem_cfr_result/cfr_model', env, 1, device)
    custom_lp_agent = CustomAgent()
    random_agent = RandomAgent(num_actions=env.num_actions)

    agents = [custom_lp_agent, dqn_agent, cfr_agent]
    agent_names = ['Custom LP', 'DQN', 'CFR']

    # we have to swap evaluation sequence for player 1 vs player 2 
    # start with baseline eval with random agent 

    print("- BASELINE EVALUATION -\n")
    for i, agent in enumerate(agents):
        print(f"[Evaluating] Player 1: {agent_names[i]} vs Player 2: Random")
        env.set_agents([agent, random_agent])
        rewards = tournament(env, args.num_games)
        print("rewards", rewards)
        for position, reward in enumerate(rewards):
            print(f"player {position + 1}, {reward}")

        # the other way around
        print(f"[Evaluating] Player 1: Random vs Player 2: {agent_names[i]}")
        env.set_agents([random_agent, agent])
        rewards = tournament(env, args.num_games)
        print("rewards", rewards)
        for position, reward in enumerate(rewards):
            print(f"player {position + 1}, {reward}")

        print("")

    # now evaluate among the complex agents
    print("- PAIRWISE EVALUATION -\n")

    for i in range(len(agents)):
        if i == len(agents) - 1:
            break
        for j in range(i+1, len(agents)):

            agent_1 = agents[i]    
            agent_2 = agents[i+1]

            print(f"[Evaluating] Player 1: {agent_names[i]} vs Player 2: {agent_names[j]}")
            env.set_agents([agent_1, agent_2])
            rewards = tournament(env, args.num_games)
            print("rewards", rewards)
            for position, reward in enumerate(rewards):
                print(f"player {position + 1}, {reward}")

            # the other way around
            print(f"[Evaluating] Player 1: {agent_names[j]} vs Player 2: {agent_names[i]}")
            env.set_agents([agent_2, agent_1])
            rewards = tournament(env, args.num_games)
            print("rewards", rewards)
            for position, reward in enumerate(rewards):
                print(f"player {position + 1}, {reward}")

            print("")





if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    # parser.add_argument(
    #     '--env',
    #     type=str,
    #     default='leduc-onebet',
    #     choices=[
    #         'blackjack',
    #         'leduc-holdem',
    #         'limit-holdem',
    #         'doudizhu',
    #         'mahjong',
    #         'no-limit-holdem',
    #         'uno',
    #         'gin-rummy',
    #     ],
    # )
    # parser.add_argument(
    #     '--models',
    #     nargs='*',
    #     default=[
    #         'experiments/leduc_holdem_dqn_result/model.pth',
    #         'experiments/leduc_holdem_cfr_result/cfr_model',
    #     ],
    # )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=10000,
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)

