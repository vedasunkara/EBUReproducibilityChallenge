import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt

import queue
import gzip



from DQN import DQNSolver

from DQN_EBU import DQNSolverEBU

#by https://github.com/gsurma

# from scores.score_logger import ScoreLogger

GAMMA = 0.9
LEARNING_RATE = 1e-3 

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0


class Maze_rand_stochastic(object):
  class observation_space(object):
    def __init__(self):
      self.shape = [28, 28, 2]

  class action_space(object):
    def __init__(self):
      self.n = 4

  def __init__(self, height = 10, width = 10, wall = np.zeros((10,10)), opt_len = 10 + 10 - 2):
    if height < 0 or height > 10 or width < 0 or width > 10:
      raise Exception('Height and width must be in [0, 10].')

    self.height = height
    self.width = width
    self.wall = wall
    self.opt_len = opt_len
    
    self.observation_space = Maze_rand_stochastic.observation_space()
    self.action_space = Maze_rand_stochastic.action_space()

    self.images,self.labels = get_data("MNIST/train-images-idx3-ubyte.gz","MNIST/train-labels-idx1-ubyte.gz",55000)
    
    def _get_idx(labels, idx_len):
      idx = []
      for i in range(10):
        idx.append(np.where(labels == i)[0][:idx_len])
      return np.stack(idx)
   
    self.idx = _get_idx(self.labels, 4963) #4987)
  def _get_observation(self, state):
        idxs = self.idx[(state, np.random.randint(self.idx.shape[1], size = 2))]
        return np.transpose(self.images[idxs, :].reshape(-1, 28, 28), (1, 2, 0))

  def reset(self):
    self.state = np.array([0, 0])
    return self._get_observation(self.state)

  def render(self):
    position = np.zeros((10,10))
    position[self.state[0]][self.state[1]] = 10
    plt.imshow(self.wall+position)
    plt.pause(.01)
    plt.draw()

  def step(self, action,use_randomness=False):
    if 1: 
      temp_state = np.array([self.state[0],self.state[1]])
      stochasticity = 0 if not use_randomness else np.random.rand()
      
      if action == 0:
        if stochasticity < 0.8:
          temp_state[0] += 1
        elif stochasticity < 0.9:
          temp_state[1] += 1
        else:
          temp_state[1] -= 1
          
      elif action == 1:
        if stochasticity < 0.8:
          temp_state[0] -= 1
        elif stochasticity < 0.9:
          temp_state[1] += 1
        else:
          temp_state[1] -= 1
        
      elif action == 2:
        if stochasticity < 0.8:
          temp_state[1] -= 1
        elif stochasticity < 0.9:
          temp_state[0] += 1
        else:
          temp_state[0] -= 1
        
      elif action == 3:
        if stochasticity < 0.8:
          temp_state[1] += 1
        elif stochasticity < 0.9:
          temp_state[0] += 1
        else:
          temp_state[0] -= 1
        
      else:
        raise ValueError('Action should be one of 0, 1, 2, 3.')
    #print temp_state, self.state
    temp_state = np.clip(temp_state, 0, [self.height - 1, self.width - 1]) # if the agent crashes into outer wall
    
    if self.wall[temp_state[0],temp_state[1]] == 1: #if the agent crashes into inner wall
      temp_state = self.state
      
    if np.array_equal(self.state, temp_state):
      reward = -1.0; done = False
    else:
      self.state = temp_state
      reward = 0.0; done = False

    if self.state[0] == self.height - 1 and self.state[1] == self.width - 1:
      reward = 1000.0
      done = True
    return self._get_observation(self.state), reward, done, None


def get_data(inputs_file_path, labels_file_path, num_examples):
    """
    Takes in an inputs file path and labels file path, unzips both files, 
    normalizes the inputs, and returns (NumPy array of inputs, NumPy 
    array of labels). Read the data of the file into a buffer and use 
    np.frombuffer to turn the data into a NumPy array. Keep in mind that 
    each file has a header of a certain size. This method should be called
    within the main function of the model.py file to get BOTH the train and
    test data. If you change this method and/or write up separate methods for 
    both train and test data, we will deduct points.
    :param inputs_file_path: file path for inputs, something like 
    'MNIST_data/t10k-images-idx3-ubyte.gz'
    :param labels_file_path: file path for labels, something like 
    'MNIST_data/t10k-labels-idx1-ubyte.gz'
    :param num_examples: used to read from the bytestream into a buffer. Rather 
    than hardcoding a number to read from the bytestream, keep in mind that each image
    (example) is 28 * 28, with a header of a certain number.
    :return: NumPy array of inputs as float32 and labels as int8
    """

    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_examples)
        data = np.frombuffer(buf, dtype=np.uint8) / 255.0
        inputs = data.reshape(num_examples, 784)

    with open(labels_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_examples)
        labels = np.frombuffer(buf, dtype=np.uint8)

    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.int8)


def get_adjacent(x,y,maze):

    up = (x,min(9,y+1))
    down =  (x,max(0,y-1))
    left = (max(x-1,0),y)
    right = (min(x+1,9),y)

    return [e for e in [up,down,left,right] if e != (x,y) and maze[e[0]][e[1]] == 0]


def astar(maze):
    # exploredCost = {}
    explored = set()
    frontier = queue.PriorityQueue()
    first = (0,0)
    frontier.put((0,(first,0,"Start",0)))
    #this tuple contains the state,the cost,and the previous
    while frontier.empty() == False:
        nextState = frontier.get()[1]
        if nextState[0] not in explored:
            explored.add(nextState[0])
            # exploredCost[nextState[0]] = nextState[1]

            if nextState[0][0] == 9 and nextState[0][1] == 9:
                return nextState[1]
            else:
                #print nextState[0], nextState[1]
                successors = get_adjacent(nextState[0][0],nextState[0][1],maze)
                for state in successors:
                    cost = 1 + nextState[1] + abs(state[0]-9)+abs(state[1]-9)
                    costToGetHere = 1 + nextState[1]
                    if state not in explored:
                        frontier.put((cost,(state,costToGetHere,nextState[0],cost)))
    return -1




def mnist_maze(wall_density,EBU=False,stochastic=False,display=False,trials=0):
    trials = trials #37 if EBU else 0 #stochastic == False else 0
    MEMORY_SIZE =  170 if EBU else 30000
    while trials < 50:
      current_data = []
      wall = (np.random.random_sample((10,10)) < wall_density).astype(int)
      wall[0][0] = 0
      wall[9][9] = 0
      best_length = astar(wall)

      if best_length == -1:
        continue
      trials+=1


      print("best length:",best_length)
      env = Maze_rand_stochastic(height = 10, width = 10, wall = wall, opt_len = best_length)

      observation_space = env.observation_space.shape[0]
      action_space = env.action_space.n
      dqn_solver = DQNSolver(observation_space, action_space,MEMORY_SIZE,GAMMA) if EBU == False else DQNSolverEBU(observation_space, action_space,MEMORY_SIZE,GAMMA) 
      dqn_solver.old_network.model.set_weights(dqn_solver.network.model.get_weights()) 


      run_lengths = []
      state = env.reset()

      run = 0
      total_steps = 0
      while total_steps < 200000:
          # break
          print(dqn_solver.memory_size,stochastic)
          run += 1
          state = env.reset()

          state = np.reshape(np.transpose(state,[2,0,1]),[-1,2,28,28])
          step = 0
          if EBU:
          	dqn_solver.add_episode()

          while step < 1000:


              if total_steps % 2000 == 0:
                  dqn_solver.old_network.model.set_weights(dqn_solver.network.model.get_weights()) 
              total_steps+=1
              step += 1
              dqn_solver.exploration_rate = (1/(200000**2))*((total_steps-200000)**2)

              action = dqn_solver.act(tf.convert_to_tensor(state,dtype=tf.float32))
              state_next, reward, terminal, info = env.step(action,use_randomness=stochastic)
              if display == True:
                if run > 3000:
                    if run % 500 == 0:
                       env.render()

              state_next = np.reshape(np.transpose(state_next,[2,0,1]),[-1,2,28,28])

              dqn_solver.remember(tf.convert_to_tensor(state,dtype=tf.float32), action, reward, tf.convert_to_tensor(state_next,dtype=tf.float32), terminal)

              if total_steps % 50 == 0:
                  dqn_solver.experience_replay()

              state = state_next
              if terminal:
                if run % 50 == 0:
                  print(trials,"Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step/best_length))
                break

          current_data.append([total_steps,step/best_length])

      np.save("DATA/EBU={},RAND={},WALL={},TRIAL={}".format(EBU,stochastic,wall_density,trials)+"_results.npy",current_data)

if __name__ == "__main__":
    #mnist_maze(0.5,EBU=True,stochastic=True,display=False)
    # mnist_maze(0.5,EBU=True,stochastic=True,display=False)
    #mnist_maze(0.5,EBU=False,stochastic=False,display=False,trials=12)
    #mnist_maze(0.2,EBU=False,stochastic=False,display=False,trials=0)
    mnist_maze(0.5,EBU=False,stochastic=False,display=False,trials=26)
