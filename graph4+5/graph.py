import matplotlib.pyplot as plt
import numpy as np

def get_rewards_and_q(file):
	lines = open(file).readlines()
	rewards = []
	q_values = []
	for line in lines:
		reward, q_value = line.strip().split()
		reward = float(reward)
		q_value = float(q_value)
		# if q_value < 0 or reward > 90:
		# 	continue
		rewards.append(reward)
		q_values.append(q_value)
	return rewards, q_values

dqn_rewards, dqn_q_values = get_rewards_and_q("mean_q_test_score_dqn.csv")
ebu_rewards, ebu_q_values = get_rewards_and_q("mean_q_test_score_ebu_1.csv")
ebu_n = len(ebu_rewards)
dqn_rewards = dqn_rewards[:ebu_n]
dqn_q_values = dqn_q_values[:ebu_n]

# plt.xticks(np.arange(0, 200, 50))
# plt.yticks(np.arange(0, 10, 2))
# plt.axis([0, 10, 0, 200])
plt.figure(figsize=(10,7))
fig, ax = plt.subplots()
plt.scatter(dqn_rewards, dqn_q_values, marker='o', label="DQN")
plt.scatter(ebu_rewards, ebu_q_values, marker='x', label="EBU, beta=1.0")
plt.xticks(range(0, 300, 50))
plt.yticks(range(0, 20, 2))
plt.ylim(0, 10)
plt.xlim(0, 200)
ax.legend()
ax.grid(True)
plt.xlabel("Test Episode Score")
plt.ylabel("Mean Q Values Across All Test Episodes")
plt.show()