import numpy as np
import copy

from config import *

def is_legal(x,y):
	return (x>=1)&(x<=grid_size)&(y>=1)&(y<=grid_size)

class Surrounding(object):
	def __init__(self, n_agent):
		super(Surrounding, self).__init__()
		self.n_agent = n_agent
		self.n_action = 5

		self.grid = self.build_env()
		self.agents = []
		for i in range(self.n_agent):
			self.agents.append([np.random.randint(0,30)+1,np.random.randint(0,30)+1])

		self.foods = []
		for i in range(self.n_agent):
			self.foods.append(self.max_food)

		self.n_resource = 8
		self.resource = []
		self.resource_pos = []
		for i in range(self.n_resource):
			self.resource_pos.append([np.random.randint(0,30)+1,np.random.randint(0,30)+1])
			self.resource.append(np.random.randint(100,120))
		
		self.steps = 0
		self.len_obs = observation_length

	def reset(self):

		self.grid = self.build_env()

		self.ants = []
		for i in range(self.n_agent):
			self.ants.append([np.random.randint(0,30)+1,np.random.randint(0,30)+1])

		self.foods = []
		for i in range(self.n_agent):
			self.foods.append(self.max_food)

		self.resource = []
		self.resource_pos = []
		for i in range(self.n_resource):
			self.resource_pos.append([np.random.randint(0,30)+1,np.random.randint(0,30)+1])
			self.resource.append(np.random.randint(100,120))

		return self.get_obs(), self.get_adj()

	def build_env(self):

		grid = np.zeros((grid_size, grid_size))
		for i in range(grid_size):
			grid[0][i] = -1
			grid[i][0] = -1
			grid[grid_size - 1][i] = -1
			grid[i][grid_size - 1] = -1

		return grid