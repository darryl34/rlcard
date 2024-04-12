''' An example of evluating the trained models in RLCard
'''
import os
import sys
import argparse
import numpy as np
import pickle

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
    
    random_seed = 42
    # Seed numpy, torch, random
    set_seed(random_seed)

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

    reward_dict_baseline = {
            'Custom LP_random': {'p1':[], 'p2':[]},
            'random_Custom LP': {'p1':[], 'p2':[]},
            'DQN_random': {'p1':[], 'p2':[]},
            'random_DQN': {'p1':[], 'p2':[]},
            'CFR_random': {'p1':[], 'p2':[]},
            'random_CFR': {'p1':[], 'p2':[]},
        }

    num_tournaments_baseline = 1000

    print("- BASELINE EVALUATION -\n")
    for i in range(num_tournaments_baseline):
        for i, agent in enumerate(agents):
            print(f"[Evaluating] Player 1: {agent_names[i]} vs Player 2: Random")
            env.set_agents([agent, random_agent])
            rewards = tournament(env, args.num_games)
            print("rewards", rewards)
            for position, reward in enumerate(rewards):
                print(f"player {position + 1}, {reward}")

            reward_dict_baseline[f"{agent_names[i]}_random"]['p1'].append(rewards[0])
            reward_dict_baseline[f"{agent_names[i]}_random"]['p2'].append(rewards[1])

            # the other way around
            print(f"[Evaluating] Player 1: Random vs Player 2: {agent_names[i]}")
            env.set_agents([random_agent, agent])
            rewards = tournament(env, args.num_games)
            print("rewards", rewards)
            for position, reward in enumerate(rewards):
                print(f"player {position + 1}, {reward}")

            reward_dict_baseline[f"random_{agent_names[i]}"]['p1'].append(rewards[0])
            reward_dict_baseline[f"random_{agent_names[i]}"]['p2'].append(rewards[1])

            print("")

    # now evaluate among the complex agents
    print("- PAIRWISE EVALUATION -\n")

    # channge seed for each evaluation

    reward_dict = {
        'Custom LP_DQN': {'p1':[], 'p2':[]},
        'DQN_Custom LP': {'p1':[], 'p2':[]},
        'Custom LP_CFR': {'p1':[], 'p2':[]},
        'CFR_Custom LP': {'p1':[], 'p2':[]},
        'DQN_CFR': {'p1':[], 'p2':[]},
        'CFR_DQN': {'p1':[], 'p2':[]},
        'Custom LP_Custom LP': {'p1':[], 'p2':[]},
        'DQN_DQN': {'p1':[], 'p2':[]},
        'CFR_CFR': {'p1':[], 'p2':[]},
    }

    num_tournaments_comparison = 1000

    for i in range(num_tournaments_comparison):
        random_seed = np.random.randint(1, 1000)
        set_seed(random_seed)
        print(f"Seed: {random_seed}")
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

                # add to dict for p1 and p2
                reward_dict[f"{agent_names[i]}_{agent_names[j]}"]['p1'].append(rewards[0])
                reward_dict[f"{agent_names[i]}_{agent_names[j]}"]['p2'].append(rewards[1])


                # the other way around
                print(f"[Evaluating] Player 1: {agent_names[j]} vs Player 2: {agent_names[i]}")
                env.set_agents([agent_2, agent_1])
                rewards = tournament(env, args.num_games)
                print("rewards", rewards)
                for position, reward in enumerate(rewards):
                    print(f"player {position + 1}, {reward}")

                                # add to dict for p1 and p2
                reward_dict[f"{agent_names[j]}_{agent_names[i]}"]['p1'].append(rewards[0])
                reward_dict[f"{agent_names[j]}_{agent_names[i]}"]['p2'].append(rewards[1])

                print("")

        # self play 
        for i in range(len(agents)):
            agent_1 = agents[i]
            agent_2 = agents[i]
            print(f"[Evaluating] Player 1: {agent_names[i]} vs Player 2: {agent_names[i]}")
            env.set_agents([agent_1, agent_2])
            rewards = tournament(env, args.num_games)
            print("rewards", rewards)
            for position, reward in enumerate(rewards):
                print(f"player {position + 1}, {reward}")

            reward_dict[f"{agent_names[i]}_{agent_names[i]}"]['p1'].append(rewards[0])
            reward_dict[f"{agent_names[i]}_{agent_names[i]}"]['p2'].append(rewards[1])

    for key, value in reward_dict_baseline.items():
        print(f"{key} p1: {np.mean(value['p1'])}, p2: {np.mean(value['p2'])}")

    with open('baseline_result_dict.pkl', 'wb') as f:
        pickle.dump(reward_dict_baseline, f)

    print(reward_dict)

    print("final results")
    for key, value in reward_dict.items():
        print(f"{key} p1: {np.mean(value['p1'])}, p2: {np.mean(value['p2'])}")


    with open('eval_result_dict.pkl', 'wb') as f:
        pickle.dump(reward_dict, f)

    print("Results saved to eval_result_dict.pkl")



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
        default=40,
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)

