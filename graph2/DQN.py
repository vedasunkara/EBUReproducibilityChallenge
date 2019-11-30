from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow as tf
import random

import numpy as np

from model import DQNetwork 

class DQNSolver:
    def __init__(self, observation_space, action_space,memory_size,gamma):
        self.observation_space = observation_space
        self.gamma = gamma
        self.action_space = action_space
        self.memory_size = memory_size
        self.action_space = action_space
        self.memory = []
        self.num_transitions = -1
        self.network = DQNetwork(action_space,[1,28,28])
        self.old_network  = DQNetwork(action_space,[1,28,28])
        self.network.model.build()
        self.old_network.model.build()
   

    def remember(self, state, action, reward, next_state, done):
        self.num_transitions+=1
        if self.num_transitions < self.memory_size :
          self.memory.append((state, action, reward, next_state, done))
        else:
           self.memory[self.num_transitions % self.memory_size ] = (state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values =  self.network.model(state).numpy()
        return np.argmax(q_values[0])

    def experience_replay(self):
          episode = random.sample(self.memory, min(350,self.num_transitions+1))
          T=len(episode)
          episode = np.array(episode)          
          
          actions = episode[:,1]
          next_rewards = episode[:,2]
          next_states = np.squeeze(np.stack(episode[:,3]),axis=1)


          cur_states =  np.squeeze(np.stack(episode[:,0]),axis=1)
          q_tilde_temp =  self.old_network.model(tf.convert_to_tensor(next_states,tf.float32)) #self.old_model(next_states)
          q_tilde = q_tilde_temp.numpy()

          y = np.zeros(T)
          y[-1] = next_rewards[-1]

          with tf.GradientTape() as tape:
              q_values = self.network.model(cur_states)
              QA_values = tf.gather(tf.reshape(q_values,[-1]),tf.convert_to_tensor(4*np.arange(len(actions))+actions, dtype=tf.int32))

              next_QA_values = np.max(q_tilde,axis=-1)
              discounted_rewards =np.array(next_rewards,dtype=np.float32)+ self.gamma * next_QA_values
              discounted_rewards = tf.convert_to_tensor(discounted_rewards,dtype=tf.float32)
              losses = self.network.loss(QA_values,discounted_rewards)

          gradients = tape.gradient(losses, self.network.model.trainable_variables)
          self.network.optimizer.apply_gradients(zip(gradients, self.network.model.trainable_variables))