from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow as tf
import random

import numpy as np

from model import DQNetwork 

class DQNSolverEBU:
    def __init__(self, observation_space, action_space,memory_size,gamma):
        self.gamma = gamma
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory_size = memory_size
        self.action_space = action_space
        self.beta = 1
        self.episode_memory = []
        self.num_episodes = -1
        self.num_transitions = -1
        self.network = DQNetwork(action_space,[1,28,28])
        self.old_network  = DQNetwork(action_space,[1,28,28])
        self.network.model.build()
        self.old_network.model.build()

    def add_episode(self):
        self.num_episodes+=1
        if self.num_episodes < self.memory_size:
          self.episode_memory.append([])
        else:
           self.episode_memory[self.num_episodes % self.memory_size] = []
    
    def remember(self, state, action, reward, next_state, done):
        self.episode_memory[self.num_episodes % self.memory_size].append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values =  self.network.model(state).numpy()
        return np.argmax(q_values[0])

    def experience_replay(self):
          
          with tf.GradientTape() as tape: 
            batches_seen = 0
            losses = tf.convert_to_tensor(0,dtype=tf.float64)
            seen = 0
            
            while batches_seen < 350:

              episode = random.sample(self.episode_memory, 1)[0]

              T=len(episode)

              if batches_seen + T > 350:
                T = 350 - batches_seen

              batches_seen+=T

              # print(T)

              episode = np.array(episode[-T:])      

              actions = episode[:,1]
              next_rewards = episode[:,2]

              next_states = np.squeeze(np.stack(episode[:,3]),axis=1)

              cur_states =  np.squeeze(np.stack(episode[:,0]),axis=1)
              q_tilde_temp =  self.old_network.model(tf.convert_to_tensor(next_states,tf.float32)) #self.old_model(next_states)
              q_tilde = q_tilde_temp.numpy()

              y = np.zeros(T)
              y[-1] = next_rewards[-1]

              
              for k in range(T-2,0,-1):
                cur_action = actions[k]
                q_tilde[k][cur_action] = self.beta * y[k+1] + (1-self.beta) * q_tilde[k][cur_action]
                y[k] = next_rewards[k] + self.gamma * np.max(q_tilde[k,])
              q_values = self.network.model(cur_states)

              QA_values = tf.gather(tf.reshape(q_values,[-1]),tf.convert_to_tensor(4*np.arange(len(actions))+actions, dtype=tf.int32))

              curr_loss = self.network.loss(QA_values, y)
              losses += curr_loss*T #*tf.convert_to_tensor(T,dtype=tf.float64)

            losses = tf.math.divide(losses,batches_seen)
          gradients = tape.gradient(losses, self.network.model.trainable_variables)
          self.network.optimizer.apply_gradients(zip(gradients, self.network.model.trainable_variables))
