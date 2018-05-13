#!/usr/bin/python 3.5
# -*-coding:utf-8:-*-

'''
Model-free Temporal Difference
Author: Jing Wang 
'''

import monte_carlo
import random
import mdp_dp_solver
import math
from datetime import datetime

def greedy(mdp, Q, state):
	'''
	greedy search method
	'''
	legalActions = [a[1] for a in mdp.getActions(state)]

	bestAction = max([(Q[(state, action)], action) for action in legalActions])[1]

	return bestAction

def sarsa(mdp, Q, state, alpha):
	'''
	Sarsa method to improve policies
	''' 
	tolerance = 1e-5
	maxCnt = 100 
	cnt = 0
	while not mdp.isEnd(state) and cnt <= maxCnt:
		action = monte_carlo.epsilonGreedy(mdp, Q, state)
		newState, reward = mdp.succAndReward(state, action)
		newAction = monte_carlo.epsilonGreedy(mdp, Q, newState)

		prevQ = Q

		# update Q
		Q[(state, action)] = Q[(state, action)] + \
				alpha * (reward + mdp.gamma * Q[(newState, newAction)] - Q[(state, action)])

		state = newState
		action = monte_carlo.epsilonGreedy(mdp, Q, state)

		# if max([abs(Q[(state, action)] - prevQ[(state, action)]) for state, action in Q.keys()]) <= tolerance:
		# 	break
		cnt += 1
	return Q

def tdSolver(mdp, method = 'sarsa', maxIter = 10000):
	'''
	temporal difference solver
	'''
	# initialize q function
	Q = {}
	E = {}
	for state in mdp.states():
		for _, action in mdp.getActions(state):
			Q[(state, action)] = random.random()
			E[(state, action)] = 0.

	# loop state
	# for state in mdp.states():
	for _ in xrange(maxIter):
		# initialize state
		state = random.choice(mdp.states())
		alpha = 0.1
		if method == 'sarsa':
			Q = sarsa(mdp, Q, state, alpha)
		elif method == 'Qlearning':
			Q = Qlearning(mdp, Q, state, alpha)
		elif method == 'sarsa_lambda':
			lamb = 0.1
			Q, E = sarsaLambda(mdp, Q, E, state, alpha, lamb)

	pi = {}
	for state in mdp.states():
		action = max([(Q[(state, action)], action) for _, action in mdp.getActions(state)])[1]
		pi[state] = action
	return pi	


def Qlearning(mdp, Q, state, alpha):
	'''
	Q learning method
	'''

	maxCnt = 100 
	cnt = 0
	while not mdp.isEnd(state) and cnt <= maxCnt:
		action = monte_carlo.epsilonGreedy(mdp, Q, state)
		# action = boltzPolicy(mdp, Q, state, beta = 5)
		newState, reward = mdp.succAndReward(state, action)

		def greedyQ(Q, state):
			return max([Q[(state, action)] for _, action in mdp.getActions(state)])

		Q[(state, action)] = Q[(state, action)] + \
			alpha * (reward + mdp.gamma * greedyQ(Q, newState) - Q[(state, action)])

		state = newState
		action = monte_carlo.epsilonGreedy(mdp, Q, state)

		cnt += 1
	return Q

def boltzPolicy(mdp, Q, state, beta):
	'''
	boltz policy search
	'''
	prob = []
	actions = mdp.getActions(state)
	actions = [i[1] for i in actions]

	for action in actions:
		q = Q[(state, action)]
		prob.append(math.exp(q * beta))
	prob = [i / sum(prob) for i in prob]

	r = random.random()
	cumulativeProb = 0.0
	for i in range(len(actions)):
		cumulativeProb += prob[i]
		if cumulativeProb >= r:
			return actions[i]
	return actions[-1]

def sarsaLambda(mdp, Q, E, state, alpha, lamb):
	'''
	sarsa lambda method
	'''
	maxCnt = 100
	cnt = 0

	while not mdp.isEnd(state) and cnt <= maxCnt:
		action = monte_carlo.epsilonGreedy(mdp, Q, state)
		newState, reward = mdp.succAndReward(state, action)
		newAction = monte_carlo.epsilonGreedy(mdp, Q, newState)

		prevQ = Q

		# update Q
		delta = reward + mdp.gamma * Q[(newState, newAction)] - Q[(state, action)]
		E[(state, action)] = E[(state, action)] + 1

		for state in mdp.states():
			for _, action in mdp.getActions(state):
				Q[(state, action)] = Q[(state, action)] + alpha * delta * E[(state, action)]
				E[(state, action)] = mdp.gamma * lamb * E[(state, action)]

		state = newState
		action = monte_carlo.epsilonGreedy(mdp, Q, state)

		# print(Q)
		# for key, val in E.items():
		# 	if val != 0:
		# 		print(key)
		# raise

		# if max([abs(Q[(state, action)] - prevQ[(state, action)]) for state, action in Q.keys()]) <= tolerance:
		# 	break
		cnt += 1

	return Q, E


def TD(alpha, gamma, states, rewards):
	'''
	temporal difference policy evaluation
	'''

	V = {}
	for state in states:
		# initialize value function without zero
		# because it makes difference
		V[state] = random.random()

	## update value function based on iteration
	maxIter = len(states)
	for t in range(maxIter):
		stepNum = len(states[t])
		for step in range(stepNum):
			state = states[t][step]
			reward = rewards[t][step]

			if step < stepNum - 1: # not the end state
				newState = states[t][step+1]
				newV = V[newState]
			else:
				newV = 0

			# update value at state
			V[state] = V[state] + alpha * (reward + gamma * newV - V[state])
	
	return V

if __name__ == '__main__':
	mdp = mdp_dp_solver.MazeMDP(5)
	start = datetime.now()
	pi = tdSolver(mdp, method = 'Qlearning', maxIter = 10000)
	end = datetime.now()
	print('Time spent: ', (end-start).total_seconds())
	state = mdp.startState()
	print(pi)
	# raise
	while not mdp.isEnd(state):
		action = pi[state]
		newState, _ = mdp.succAndReward(state, action)
		print('State {} -> New State {} by Action {}'.format(state, newState, action))
		state = newState
