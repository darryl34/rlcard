import pickle
import matplotlib.pyplot as plt

with open('eval_result_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)


# create graphs for: LP vs CFR, DQN vs CFR (one graph), the other graph flip the order

plt.plot(loaded_dict['Custom LP_CFR']['p1'], label="LP (p1)")
plt.plot(loaded_dict['DQN_CFR']['p1'], label="DQN (p1)")
plt.title("LP and DQN Rewards against (player 2) CFR")
plt.xlabel('Tournament Number')
plt.ylabel('Avg Rewards')
plt.legend()
plt.show()


plt.plot(loaded_dict['CFR_Custom LP']['p2'], label="LP (p2)")
plt.plot(loaded_dict['CFR_DQN']['p2'], label="DQN (p2)")
plt.title("LP and DQN Rewards against (player 1) CFR")
plt.xlabel('Tournament Number')
plt.ylabel('Avg Rewards')
plt.legend()
plt.show()

