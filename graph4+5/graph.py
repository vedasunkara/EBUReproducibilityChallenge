import matplotlib.pyplot as plt
import numpy as np
lines = open("mean_q_test_score.csv").readlines()
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

print(np.mean(rewards))
# plt.xticks(np.arange(0, 200, 50))
# plt.yticks(np.arange(0, 10, 2))
# plt.axis([0, 10, 0, 200])
plt.figure(figsize=(10,7))
plt.scatter(rewards, q_values)
plt.xticks(range(0, 300, 50))
plt.yticks(range(0, 20, 2))
plt.ylim(0, 10)
plt.xlim(0, 200)

plt.xlabel("Test Episode Score")
plt.ylabel("Mean Q Values Across All Test Episodes")
# plt.show()