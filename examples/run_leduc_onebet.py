import os
import sys

sys.path.insert(0, os.getcwd())
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard import models
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

# Make training environment
env = rlcard.make('leduc-onebet', config={'seed': 0})

# set up agents
dqn_agent = DQNAgent(num_actions=env.num_actions,
                     state_shape=env.state_shape[0],
                     mlp_layers=[64,64],
                     device=get_device())
rand_agent = RandomAgent(num_actions=env.num_actions)

cfr_agent = models.load('leduc-holdem-cfr').agents[0]

env.set_agents([dqn_agent, rand_agent])

# set up evaluation environment
eval_env = rlcard.make('leduc-onebet', config={'seed': 0})
eval_env.set_agents([dqn_agent, cfr_agent])

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
                tournament(eval_env, eval_games)[0]
            )

    csv_path, fig_path = logger.csv_path, logger.fig_path

plot_curve(csv_path, fig_path, 'DQN on one step Leduc Holdem against CFR')
