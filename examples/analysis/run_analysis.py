import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('eval_result_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

"""For losses"""

# LP
lp_1_loss = [val for val in loaded_dict['Custom LP_CFR']['p1'] if val < 0]
lp_2_loss = [val for val in loaded_dict['CFR_Custom LP']['p2'] if val < 0]
lp_3_loss = [val for val in loaded_dict['Custom LP_DQN']['p1'] if val < 0]
lp_4_loss = [val for val in loaded_dict['DQN_Custom LP']['p2'] if val < 0]
mean_lp_loss= np.mean(lp_1_loss + lp_2_loss + lp_3_loss + lp_4_loss)
lp_loss_length = len(lp_1_loss + lp_2_loss + lp_3_loss + lp_4_loss)
print(f"Mean LP loss: {mean_lp_loss}")
print(f"LP loss num: {lp_loss_length}")
print(f"toal num: {len(loaded_dict['Custom LP_CFR']['p1'])}")
print(f"LP loss percentage: {lp_loss_length/(len(loaded_dict['Custom LP_CFR']['p1'] * 4)) * 100}")

# DQN
dqn_1_loss = [val for val in loaded_dict['DQN_Custom LP']['p1'] if val < 0]
dqn_2_loss = [val for val in loaded_dict['Custom LP_DQN']['p2'] if val < 0]
dqn_3_loss = [val for val in loaded_dict['DQN_CFR']['p1'] if val < 0]
dqn_4_loss = [val for val in loaded_dict['CFR_DQN']['p2'] if val < 0]
mean_dqn_loss = np.mean(dqn_1_loss + dqn_2_loss + dqn_3_loss + dqn_4_loss)
dqn_loss_length = len(dqn_1_loss + dqn_2_loss + dqn_3_loss + dqn_4_loss)
print(f"Mean DQN loss: {mean_dqn_loss}")
print(f"DQN loss num: {dqn_loss_length}")
print(f"DQN loss percentage: {dqn_loss_length/len(loaded_dict['DQN_CFR']['p1']*4) * 100}")

# CFR
cfr_1_loss = [val for val in loaded_dict['CFR_DQN']['p1'] if val < 0]
cfr_2_loss = [val for val in loaded_dict['DQN_CFR']['p2'] if val < 0]
cfr_3_loss = [val for val in loaded_dict['CFR_Custom LP']['p1'] if val < 0]
cfr_4_loss = [val for val in loaded_dict['Custom LP_CFR']['p2'] if val < 0]
cfr_loss_length = len(cfr_1_loss + cfr_2_loss + cfr_3_loss + cfr_4_loss)
mean_cfr_loss= np.mean(cfr_1_loss + cfr_2_loss + cfr_3_loss + cfr_4_loss)
print(f"Mean CFR loss: {mean_cfr_loss}")
print(f"CFR loss num: {cfr_loss_length}")
print(f"CFR loss percentage: {cfr_loss_length/len(loaded_dict['CFR_DQN']['p1']*4) * 100}")


"""For wins"""

# LP
lp_1_wins = [val for val in loaded_dict['Custom LP_CFR']['p1'] if val >= 0]
lp_2_wins = [val for val in loaded_dict['CFR_Custom LP']['p2'] if val >= 0]
lp_3_wins = [val for val in loaded_dict['Custom LP_DQN']['p1'] if val >= 0]
lp_4_wins = [val for val in loaded_dict['DQN_Custom LP']['p2'] if val >= 0]
num_lp_wins = len(lp_1_wins + lp_2_wins + lp_3_wins + lp_4_wins)
mean_lp_wins= np.mean(lp_1_wins + lp_2_wins + lp_3_wins + lp_4_wins)
print(f"Mean LP wins: {mean_lp_wins}")
print(f"LP wins num: {num_lp_wins}")
print(f"LP wins percentage: {num_lp_wins/(len(loaded_dict['Custom LP_CFR']['p1'])*4) * 100}")

# DQN
dqn_1_wins = [val for val in loaded_dict['DQN_Custom LP']['p1'] if val >= 0]
dqn_2_wins = [val for val in loaded_dict['Custom LP_DQN']['p2'] if val >= 0]
dqn_3_wins = [val for val in loaded_dict['DQN_CFR']['p1'] if val >= 0]
dqn_4_wins = [val for val in loaded_dict['CFR_DQN']['p2'] if val >= 0]
num_dqn_wins = len(dqn_1_wins + dqn_2_wins + dqn_3_wins + dqn_4_wins)
mean_dqn_wins = np.mean(dqn_1_wins + dqn_2_wins + dqn_3_wins + dqn_4_wins)
print(f"Mean DQN wins: {mean_dqn_wins}")
print(f"DQN wins num: {num_dqn_wins}")
print(f"DQN wins percentage: {num_dqn_wins/(len(loaded_dict['DQN_CFR']['p1'])*4) * 100}")


# CFR
cfr_1_wins = [val for val in loaded_dict['CFR_DQN']['p1'] if val >= 0]
cfr_2_wins = [val for val in loaded_dict['DQN_CFR']['p2'] if val >= 0]
cfr_3_wins = [val for val in loaded_dict['CFR_Custom LP']['p1'] if val >= 0]
cfr_4_wins = [val for val in loaded_dict['Custom LP_CFR']['p2'] if val >= 0]
num_cfr_wins = len(cfr_1_wins + cfr_2_wins + cfr_3_wins + cfr_4_wins)
mean_cfr_wins= np.mean(cfr_1_wins + cfr_2_wins + cfr_3_wins + cfr_4_wins)
print(f"Mean CFR wins: {mean_cfr_wins}")
print(f"CFR wins num: {num_cfr_wins}")
print(f"CFR wins percentage: {num_cfr_wins/(len(loaded_dict['CFR_DQN']['p1'])*4) * 100}")