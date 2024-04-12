import pickle
import matplotlib.pyplot as plt

with open('baseline_result_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

# create graphs for baseline agent

for key, value in loaded_dict.items():
    p1, p2 = key.split("_")
    title = " vs ".join(key.split("_"))
    plt.plot( [x for x in range(len(value['p1']))],value['p1'], label=f"{p1} (p1)")
    plt.title(title)
    plt.xlabel('Tournament Number')
    plt.ylabel('Avg Rewards')
    plt.legend()
    plt.show()