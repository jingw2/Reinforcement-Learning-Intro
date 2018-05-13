#!/usr/bin/python 3.5
# -*-coding:utf-8:-*-

'''
Markov Decision Process Dynamic Programming Solver
Note: it is for action -> state mapping
Author: Jing Wang
'''

import numpy as np 
from copy import deepcopy
import random

class MDP(object):

	def __init__(self, N):
		self.gamma = 1
		self.N = N

	def isEnd(self, state):
		'''
		check if the state is end state

		Args:
		state

		Return:
		boolean variable
		'''
		return state == self.N

	def getActions(self, state):
		'''
		get actions and probability of actions
		'''
		result = []
		if state + 1 <= self.N:
			result.append((0.5, 'walk'))
		if state * 2 <= self.N:
			result.append((0.5, 'tram'))
		return result

	def succAndReward(self, state, action):
		if action == 'walk':
			return (state + 1, -1)
		if action == 'tram':
			return (state * 2, -2)

	def transform(self, s1, s2):
		return 1

	def states(self):
		return list(range(1, N+1))

class PolicyIteration(object):

	def __init__(self):
		self.pi = {}

	def initializePolicy(self, mdp):
		for state in mdp.states():
			actions = mdp.getActions(state)
			if len(actions) != 0:
				self.pi[state] = random.choice(actions)[1]
			else:
				self.pi[state] = None

	def policyEvaluation(self, mdp):
		'''
		evaluate current policy 

		Args:
		mdp (class object)

		Return:
		V (dict): value function, key is state, 
				value is value of each state

		'''
		epsilon = 1e-6

		## initialize value function
		V = {}
		for state in mdp.states():
			V[state] = 0

		def Value(state, action):
			newState, reward = mdp.succAndReward(state, action)
			return reward + mdp.gamma * V[newState] * mdp.transform(state, newState)

		## repeat to converge 
		while True:
			newV = {}
			for state in mdp.states(): # loop for each state
				# check end state 
				if mdp.isEnd(state): 
					newV[state] = 0
					continue

				action = self.pi[state]

				# update value function
				newV[state] = Value(state, action)

			# check tolerance
			if max([abs(newV[state] - V[state]) for state in mdp.states()]) <= epsilon:
				break
			else:
				V = newV

		return V

	def policyImprovement(self, mdp):
		'''
		improve policy
		
		Args:
		mdp (class object)

		Return:
		pi (dict): best action for each state
		'''

		for state in mdp.states():
			if mdp.isEnd(state): continue 

			V = self.policyEvaluation(mdp)

			qValues = []
			for (prob, action) in mdp.getActions(state):
				newState, reward = mdp.succAndReward(state, action)
				q = reward + mdp.gamma * V[newState] * mdp.transform(state, newState)
				qValues.append(q)

			bestIndex = qValues.index(max(qValues))
			bestAction = mdp.getActions(state)[bestIndex][1]
			self.pi[state] = bestAction

	def solve(self, mdp):
		iterCnt = 0
		stop = True
		self.initializePolicy(mdp)

		while True:
			iterCnt += 1 
			print('Iteration: ', iterCnt)
			piCopy = deepcopy(self.pi)

			# self.policyEvaluation(mdp)
			self.policyImprovement(mdp)

			for state in mdp.states():
				if piCopy[state] != self.pi[state]:
					stop = False
			if stop:
				break
			stop = True

class ValueIteration(object):

	def __init__(self):
		self.pi = {}

	def solve(self, mdp):
		iterCnt = 0
		epsilon = 1e-6
		## initialize value function
		V = {}
		for state in mdp.states():
			V[state] = 0

		def Value(state, action):
			newState, reward = mdp.succAndReward(state, action)

			return reward + mdp.gamma * V[newState] * mdp.transform(state, newState)

		while True:
			newV = {}
			iterCnt += 1 
			print('Iteration: ', iterCnt)

			for state in mdp.states():
				if mdp.isEnd(state):
					newV[state] = 0 
					self.pi[state] = None
					continue 
				newV[state] = max([Value(state, action) \
							for (prob, action) in mdp.getActions(state)])
				self.pi[state] = max([(Value(state, action), action) \
							for (prob, action) in mdp.getActions(state)], \
							key = lambda x: x[0])[1]

			# check tolerance
			if max([abs(newV[state] - V[state]) for state in mdp.states()]) <= epsilon:
				break
			else:
				V = newV

class MazeMDP(object):

	def __init__(self, size):
		'''
		Args:
		size (int): e.g. 5, which is 5 * 5 grid world
		'''
		self.size = size
		self.gamma = 0.9 
		self.walls = [(3, 0), (3, 1), (0, 2), (1, 2), (2, 4), (3, 4), (4, 4)]
		
	def isEnd(self, state):
		return state == (4, 2) 

	def startState(self):
		return (0, 0)

	def getActions(self, state):
		actions = []
		x, y = state 
		xmin, ymin = 0, 0
		cnt = 0
		xmax, ymax = self.size - 1, self.size - 1
		if x + 1 <= xmax and (x + 1, y) not in self.walls:
			cnt += 1
			actions.append((0.25, 'right'))
		if x - 1 >= xmin and (x - 1, y) not in self.walls:
			cnt += 1
			actions.append((0.25, 'left'))
		if y - 1 >= ymin and (x, y - 1) not in self.walls:
			cnt += 1
			actions.append((0.25, 'up'))
		if y + 1 <= ymax and (x, y + 1) not in self.walls:
			cnt += 1
			actions.append((0.25, 'down'))
		actions = [(1./ cnt, act) for _, act in actions]
		# random.shuffle(actions)
		return actions


	def succAndReward(self, state, action):
		newState = None
		x, y = state
		if action == 'up':
			newy = y - 1 
			if (x, newy) in self.states():
				newState = (x, newy)
			else:
				newState = (x, y)
		elif action == 'down':
			newy = y + 1
			if (x, newy) in self.states():
				newState = (x, newy)
			else:
				newState = (x, y)
		elif action == 'left':
			newx = x - 1
			if (newx, y) in self.states():
				newState = (newx, y)
			else:
				newState = (x, y)
		elif action == 'right':
			newx = x + 1 
			if (newx, y) in self.states():
				newState = (newx, y)
			else:
				newState = (x, y)

		if newState in self.walls:
			reward = -float('inf')
		elif newState == (4, 2):
			reward = 100
		else:
			reward = 0 
		return (newState, reward)

	def transform(self, s1, s2):
		return 1

	def states(self):
		result = []
		for x in range(self.size):
			for y in range(self.size):
				result.append((x, y))
		return result

if __name__ == '__main__':
	mdp = MazeMDP(5)

	## It is likely not to converge
	# print('State: ', mdp.states())
	# piter = PolicyIteration()
	# piter.solve(mdp)
	# print('policy iteration: ', piter.pi)

	viter = ValueIteration()
	viter.solve(mdp)
	print('value iteration: ', viter.pi)

	state = mdp.startState()
	while not mdp.isEnd(state):
		action = viter.pi[state]
		newState, _ = mdp.succAndReward(state, action)
		print('State {} -> New State {} by Action {}'.format(state, newState, action))
		state = newState

