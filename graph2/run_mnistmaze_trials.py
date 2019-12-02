import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
from dqn_test_new_maze import *
#from theirmaze import *
import queue
import gzip



from DQN import DQNSolver

from DQN_EBU import DQNSolverEBU

from DQN_NSTEP import DQNSolverNSTEP

#by https://github.com/gsurma

GAMMA = 0.9
LEARNING_RATE = 1e-3 

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0


def run_mnist_maze(wall_density,EBU=False,stochastic=False,display=False,trials=0,EBU_VAL=None,NSTEP=False):
    trials = trials
    MEMORY_SIZE =  170 if EBU or NSTEP else 30000
    while trials < 40:
      current_data = []
      environment = Maze(wall_density) if not stochastic else Maze_Stochastic(wall_density)
      
      best_length = environment.best_length
      print("BEST LENGTH:",best_length)
      trials+=1

      observation_space = [28,28]
      action_space = 4

      if NSTEP is False:

          dqn_solver = DQNSolver(observation_space, action_space,MEMORY_SIZE,GAMMA) if EBU == False else DQNSolverEBU(observation_space, action_space,MEMORY_SIZE,GAMMA,EBU_VAL) 
      else:
          dqn_solver = DQNSolverNSTEP(observation_space, action_space,MEMORY_SIZE,GAMMA)
      dqn_solver.old_network.model.set_weights(dqn_solver.network.model.get_weights()) 
   

      run_lengths = []

      run = 0
      total_steps = 0
      while total_steps < 200000:
          # break
          #print(dqn_solver.memory_size,stochastic)
          run += 1
          state = environment.reset()
          state = np.expand_dims(state,axis=0) #np.reshape(np.transpose(state,[2,0,1]),[-1,2,28,28])

          step = 0
          if EBU or NSTEP:
          	dqn_solver.add_episode()

          while step < 1000:


              if total_steps % 2000 == 0:
                  dqn_solver.old_network.model.set_weights(dqn_solver.network.model.get_weights()) 
              total_steps+=1
              step += 1
              dqn_solver.exploration_rate = (1/(200000**2))*((total_steps-200000)**2)

              action = dqn_solver.act(tf.convert_to_tensor(state,dtype=tf.float32))
              state_next, reward, terminal = environment.act(action)
              # print(state_next.shape)
              if display == True:
                if run >= 100:
                    if run % 100 == 0:
                       environment.render()

              state_next = np.expand_dims(state_next,axis=0) #np.reshape(np.transpose(state_next,[2,0,1]),[-1,2,28,28])
              dqn_solver.remember(tf.convert_to_tensor(state,dtype=tf.float32), action, reward, tf.convert_to_tensor(state_next,dtype=tf.float32), terminal)

              if total_steps % 50 == 0: #and not NSTEP:
                  dqn_solver.experience_replay()


              state = state_next
              if terminal:
                if run % 50 == 0:
                  print(trials,"Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step/best_length))
                break



          current_data.append([total_steps,step/best_length])

      np.save("DATA/EBU={},NSTEP={},RAND={},WALL={},TRIAL={},BETA={}".format(EBU,NSTEP,stochastic,wall_density,trials,EBU_VAL)+"_results.npy",current_data)

if __name__ == "__main__":

  #TO RUN:
  # #20% Determnistic:
    run_mnist_maze(0.2,EBU=False,stochastic=False,display=False,NSTEP=True,trials=0)
    run_mnist_maze(0.2,EBU=False,stochastic=False,display=False,trials=0)
    run_mnist_maze(0.2,EBU=True,stochastic=False,display=False,EBU_VAL=1.0,trials=0)


  # #50% Determnistic:
    run_mnist_maze(0.5,EBU=True,stochastic=False,display=False,EBU_VAL=1.0,trials=0)
    run_mnist_maze(0.5,EBU=False,stochastic=False,display=True,NSTEP=True,trials=0)
    run_mnist_maze(0.5,EBU=False,stochastic=False,display=False,trials=0)


  #50% Stochastic

    run_mnist_maze(0.5,EBU=True,stochastic=True,display=False,EBU_VAL=1.0,trials=0)
    run_mnist_maze(0.5,EBU=True,stochastic=True,display=False,EBU_VAL=0.75,trials=0)
    run_mnist_maze(0.5,EBU=True,stochastic=True,display=False,EBU_VAL=0.5,trials=0)
    run_mnist_maze(0.5,EBU=True,stochastic=True,display=False,EBU_VAL=0.25,trials=0)
    run_mnist_maze(0.5,EBU=False,stochastic=True,display=False,trials=0)
    run_mnist_maze(0.5,EBU=False,stochastic=True,display=False,NSTEP=True,trials=0)  


