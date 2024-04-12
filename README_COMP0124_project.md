# COMP0124 (Multi-agent Artificial Intelligence)

### Project title: Comparative Analysis of Strategy Performance in Simplified Poker

This code is a deliverable for Coursework 2 (research project) for COMP0124, along with a report writeup.

In our development process, we forked the RLCard Library and made changes in the library directly. These changes involved creating a custom environment for one-bet Leduc poker and creating a custom agent that uses the a Linear Programming (LP) strategy.

---

### Theoretical Analysis

As a preliminary step, we did a theoretical analysis on the LP strategy which involved analytically deriving the Nash Equilibrium in Kuhn and one-bet Leduc poker.

The jupyter notebook can be found here: 
- theoretical_analysis/Leduc_LP.ipynb



---

### High level developement steps

(1) We first had to create custom components for our one-bet Leduc Poker experiment: 

- The custom one-bet Leduc Poker environment can be found here: 
  - rlcard/envs/leduc_onebet.py
  - rlcard/games/leduc_onebet/game.py
  
- The custom LP agent can be found here:
  - rlcard/agents/custom_leduc_onebet_agent.py

(2) We trained Deep Q-learning (DQN) and Counterfactual Regret Minimization (CFR) agents for our comparisons 

- The code for training the models can be found here:
  - CFR agent: examples/run_cfr.py
  - DQN agent: examples/run_leduc_onebet.py

- The code produces models that are found here (submission includes these pre-trained models). Along with the model, csv logs and graphs are also produced to show the training of the agent:
  - CFR agent: experiments/leduc_holdem_cfr_result
  - DQN agent: experiments/leduc_holdem_dqn_result


(3) We wrote simulation and evaluation code to calculate the statistics for our pairwise agent comparisons. We output traces of agent rewards as tournaments run

- The simulation code can be found here: 
  - examples/evaluate.py

- The analysis code can be found here: 
  - examples/analysis

---

### Running the analysis 

Included in our submission are pre-trained models and pickle objects containing data from our simulation runs. 

To view important graphs and statistics run:

- examples/analysis/run_analysis.py 
- examples/analysis/run_graph_eval.py