import os
import sys
import torch
# import pytorch as torch

sys.path.insert(0, os.getcwd())
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)



# Make environment
env = rlcard.make('leduc-onebet')

# set up agents
dqn_agent = DQNAgent(num_actions=env.num_actions,
                     state_shape=env.state_shape[0],
                     mlp_layers=[64,64],
                     device=get_device())

rand_agent = RandomAgent(num_actions=env.num_actions)
env.set_agents([dqn_agent, rand_agent])


num_eps = 5000
eval_every = 100
eval_games = 1000
with Logger('leduc_holdem_dqn') as logger:
    for ep in range(num_eps):
        # generate data from the environment
        trajectories, payoffs = env.run(is_training=True)

        # reorganize data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)

        # feed transitions into agent memory, and train the agent
        # assume dqn agent always plays first
        for ts in trajectories[0]:
            dqn_agent.feed(ts)

        # evaluate the performance. Play with random agents.
        if ep % eval_every == 0:
            logger.log_performance(
                ep, 
                tournament(env, eval_games)[0]
            )

    csv_path, fig_path = logger.csv_path, logger.fig_path

plot_curve(csv_path, fig_path, 'DQN on one step Leduc Holdem')

default_path = os.path.join(os.getcwd(), "experiments/leduc_holdem_dqn_result/")
if not os.path.exists(default_path):
    os.makedirs(default_path)
save_path = os.path.join(default_path, 'model.pth')

# Assuming dqn_agent is your model or a dictionary containing your model's state
torch.save(dqn_agent, save_path)
print('Model saved in', save_path)