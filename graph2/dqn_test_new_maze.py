import tensorflow as tf
import numpy as np
import copy
import queue

class Maze(object):
	'''
	A class representing a 10 x 10 2D MNIST Maze
	'''
	def __init__(self, wall_density):
		'''

		:param wall_density: the probability of a wall at any location in the maze
		'''
		self.height = 10
		self.width = 10
		self.goal_state = np.array([9, 9])
		# self.wall_density = wall_density
		self.state = (0, 0)
		# a 2D binary array representing wall locations (1 = wall, 0 = no wall)
		
		self.walls,self.best_length = self.create_maze(wall_density)

		(train_images, train_labels), (test_images, test_labels) = \
			tf.keras.datasets.mnist.load_data()
		self.x_trains = np.array([train_images[train_labels == i] for i in range(10)])
		self.x_tests = np.array([test_images[test_labels == i] for i in range(10)])


	def create_maze(self,wall_density):
		while True:
			wall = (np.random.random_sample((self.width,self.height)) < wall_density).astype(int)
			wall[self.state[0]][self.state[1]] = 0
			wall[self.goal_state[0]][self.goal_state[1]] = 0
			best_length = self.astar(wall)
			if best_length != -1:
				return wall,best_length


	def get_adjacent(self,x,y,maze):

	    up = (x,min(9,y+1))
	    down =  (x,max(0,y-1))
	    left = (max(x-1,0),y)
	    right = (min(x+1,9),y)

	    return [e for e in [up,down,left,right] if e != (x,y) and maze[e[0]][e[1]] == 0]


	def astar(self,maze):
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
	                successors = self.get_adjacent(nextState[0][0],nextState[0][1],maze)
	                for state in successors:
	                    cost = 1 + nextState[1] + abs(state[0]-9)+abs(state[1]-9)
	                    costToGetHere = 1 + nextState[1]
	                    if state not in explored:
	                        frontier.put((cost,(state,costToGetHere,nextState[0],cost)))
	    return -1

	def get_state_mnist(self, state):
		'''

		:param state: a tuple representing the coordinates in the maze (x, y) where 0 <= x, y <= 9
		:return: a 2-element numpy array of MNIST digits corresponding to the state passed in
		'''



		x_index = np.random.choice(np.arange(self.x_trains[state[0]].shape[0]))
		y_index = np.random.choice(np.arange(self.x_trains[state[1]].shape[0]))

		# print(self.x_trains[state[0]][x_index].shape)

		state_data = np.stack([self.x_trains[state[0]][x_index],self.x_trains[state[1]][y_index]])
		# print(state_data.shape)

		return state_data

	def reset(self):
		self.state = np.array([0,0])
		self.state_mnist = self.get_state_mnist(self.state)
		return self.state_mnist

	def act(self, action):
		'''

		:param action: 0 (up), 1 (right), 2 (down), or 3 (left)
		:return: a 3-tuple containing the MNIST pair representing the state after acting,
		the reward after acting, and a boolean indicating whether the goal state has been reached
		'''
		next_state = copy.copy(self.state)
		if action == 0:
			next_state[1] = min(next_state[1] + 1, 9)
		elif action == 1:
			next_state[0] = min(next_state[0] + 1, 9)
		elif action == 2:
			next_state[1] = max(next_state[1] - 1, 0)
		elif action == 3:
			next_state[0] = max(next_state[0] - 1, 0)
		else:
			raise ValueError('Invalid Action')

		if self.walls[next_state[0]][next_state[1]] == 1:
			reward = -1.0
			reached_goal = False
		else:
			self.state = copy.copy(next_state)
			if np.array_equal(self.state, self.goal_state):
				reward = 1000.0
				reached_goal = True
			else:
				reward = 0.0
				reached_goal = False
		return self.get_state_mnist(self.state), reward, reached_goal

class Maze_Stochastic(Maze):
	'''
	A class representing the same 2D MNIST maze environment above but with stochasticity
	'''
	def __init__(self, wall_density, stochasticity=0.20):
		'''

		:param wall_density: the probability of a wall at any location in the maze
		:param stochasticity: the probability that a random action is selected when the act() method is called
		'''
		super().__init__(walls)
		self.stochasticity = stochasticity

		self.random_actions = {0:[1,3],1:[2,4],2:[1,3],3:[2,4]}

	def act(self, action):
		next_state = copy.copy(self.state)
		rand_num = np.random.rand()

		if rand_num < self.stochasticity:
			action = self.random_actions[action][rand_num <= (self.stochasticity // 2)]

		if action == 0:
			next_state[1] = min(next_state[1] + 1, 9)
		elif action == 1:
			next_state[0] = min(next_state[0] + 1, 9)
		elif action == 2:
			next_state[1] = max(next_state[1] - 1, 0)
		elif action == 3:
			next_state[0] = max(next_state[0] - 1, 0)
		else:
			raise ValueError('Invalid Action')
		if self.walls[next_state] == 1:
			reward = -1.0
			reached_goal = False
		else:
			self.state = copy.copy(next_state)
			if np.array_equal(self.state, self.goal_state):
				reward = 1000.0
				reached_goal = True
			else:
				reward = 0.0
				reached_goal = False
		return self.get_state_mnist(self.state), reward, reached_goal
