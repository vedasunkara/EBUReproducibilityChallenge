import numpy as np
import random

############################################
# Performing EBU for tabular Q-value update#
############################################

# there are four states, (s1=0, s2=1, s3=2, s4=3)
state_size = 4
# there are two actions, (l=0 or r=1)
action_size = 2
# initialize a discount factor for update
delta = 0.1

num_tests = 100

episodes = []
for i in range(num_tests):
  episode = []
  current_state = 0

  while(current_state != 3):
    action_probabilities = np.zeros((2,))
    action_probabilities[0] = 0.5
    action_probabilities[1] = 0.5

    # sample an action based on the probability distribution
    action = np.random.choice(action_size, p = action_probabilities)

    sT = current_state

    # perform the action
    if(action == 0):
      if(current_state != 0):
        current_state = current_state - 1
    if(action == 1):
      current_state = current_state + 1

    reward = 0
    if(current_state == 3):
      reward = 1

    episode.append((sT, action, reward, current_state))

  episodes.append(episode)

opt_path = np.zeros(50, dtype=np.float32)

for transitions in range(5, 50):
  # initialize the Q-table with a zero matrix
  Q = np.zeros((state_size, action_size))
  # for the base case, with zeros
  has_update = np.zeros((state_size,), dtype=np.int)

  if (opt_path[transitions-1] == 1.):
    opt_path[transitions] = 1.
  else:
    n = transitions
    while n > 0:
      ep = random.choice(episodes)
      for sars in reversed(ep):
        sT = sars[0]
        aT = sars[1]
        rT = sars[2]
        sT1 = sars[3]

        # EBU computation
        Q[sT, aT] = rT + delta * np.max(a=Q[sT1])
        # maintain updates (for base case)
        has_update[sars[0]] = 1
        n = n - 1

    eps = []
    cs = 0

    while(len(eps) <= 3):
      action_probabilities = np.zeros((2,))
      action_probabilities[0] = 0.5
      action_probabilities[1] = 0.5
      if(has_update[cs] == 1):
        action_probabilities = Q[cs]/Q[cs].sum()

      action = np.random.choice(action_size, p = action_probabilities)

      sT = current_state

      # perform the action
      if(action == 0):
        if(cs != 0):
          cs = cs - 1
      if(action == 1):
        cs = cs + 1

      reward = 0
      if(cs == 3):
        reward = 1

      eps.append((sT, action, reward, cs))

      if(len(eps) == 3):
        opt_path[transitions] = 1.
      else:
        opt_path[transitions] = 0.

#################################################################################################
# Performing Uniform Sampling for Update (mostly similar to above except for probability update)#
#################################################################################################

transitions = []
num_t = 10000

for i in range(num_t):
  current_state = 0

  while (current_state != 3):
    action_probabilities = np.zeros((2,))
    action_probabilities[0] = 0.5
    action_probabilities[1] = 0.5

    action = np.random.choice(action_size, p = action_probabilities)
    sT = current_state

    # perform the action
    if(action == 0):
      if(current_state != 0):
        current_state = current_state - 1
    if(action == 1):
      current_state = current_state + 1

    reward = 0
    if(current_state == 3):
      reward = 1

    transitions.append((sT, action, reward, current_state))

num_optimal = np.zeros(50)
# initialize the Q-table with a zero matrix
# Q = np.zeros((state_size, action_size))
# for the base case, with zeros
# has_update = np.zeros((state_size,), dtype=np.int)

avg = 1000

for i in range(50):
  for t in range(avg):
    Q = np.zeros((state_size, action_size))
    has_update = np.zeros((state_size,), dtype=np.int)
    num_iter = i
    for x in range(num_iter):
      sars = random.choice(transitions)
      sT = sars[0]
      aT = sars[1]
      rT = sars[2]
      sT1 = sars[3]

      # EBU computation
      Q[sT, aT] = rT + delta * np.max(a=Q[sT1])
      # maintain updates (for base case)
      has_update[sT] = 1

    eps = []
    cs = 0
    while(cs != 3):
      ap = np.zeros((2,))
      ap[0] = 0.5
      ap[1] = 0.5
      if(Q[cs].all() != 0):
        ap = Q[cs]/Q[cs].sum()

      action = np.argmax(Q[cs])

      sT = cs

      # perform the action
      if(action == 0):
        if(cs != 0):
          cs = cs - 1
      if(action == 1):
        cs = cs + 1

      eps.append((sT, action, reward, cs))

      if(len(eps) == 3 and cs == 3):
        num_optimal[i] = num_optimal[i] + 1.

num_optimal = np.divide(num_optimal, avg)

#################
#Generate Graphs#
#################

import matplotlib.pyplot as plt

axis = np.zeros(50)
for i in range(len(axis)):
  axis[i] = i

ebu = plt.scatter(axis, opt_path, label='episodic backward')
rnd = plt.scatter(axis, num_optimal, label='uniform sample')
plt.legend((ebu, rnd), ('Episodic Backward', 'Uniform Sample'), loc='lower right')
plt.xlabel("# of sampled transitions")
plt.ylabel("Probability of learning the optimal path")
