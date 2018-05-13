#!/usr/bin/python 3.5
# -*-coding:utf-8:-*-

'''
Model-free Monte Carlo Simulation
Author: Jing Wang 
'''

import random
import mdp_dp_solver

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
		return range(1, self.N+1)

## Monte Carlo Step 1: get random policies
def getRandomPi(mdp, num, Q, eg = True):
	'''
	Get random policies 
	states, actions, rewards 

	Args:
	mdp (class object): markov decision process 
	num (int): how many policies to generate
	Q (dict): Q functions
	eg (boolean): True for epsilon greedy to choose action
								False otherwise

	Return:
	states = []
	actions = []
	rewards = []
	'''
	states = []
	actions = []
	rewards = []
	for _ in xrange(num):
		s_n = []
		a_n = []
		r_n = []
		
		state = mdp.states[int(random.random() * len(mdp.states()))]

		## generate one experiment data
		while not mdp.isEnd(state):

			if eg:
				action = epsilonGreedy(mdp, Q, state)
			else:
				legalActions = [a[1] for a in mdp.getActions(state)]
				action = legalActions[int(random.random() * len(legalActions))]

			newState, reward = mdp.succAndReward(state, action)

			s_n.append(state)
			a_n.append(action)
			r_n.append(reward)
			state = newState

		states.append(s_n)
		actions.append(a_n)
		rewards.append(r_n)

	return states, actions, rewards 

## Step 2: evaluation
def evaluation(mdp, states, actions, rewards, epsilon = 1e-5):
	
	## value functions 
	V = {state: 0. for state in mdp.states()} # key is state, value is value
	## state count 
	stateCnt = {state : 0. for state in mdp.states()} # key is state, value is the occurrence count 

	for idx in range(len(states)):
		G = 0.0 
		num = len(states[idx])
		## backward calculate the accumulative reward 
		for index in range(num, -1, -1):
			G *= mdp.gamma 
			G += rewards[idx][index]

		## forward to calculate the accumulative reward for every state
		for index in range(num):
			state = states[idx][index]
			V[state] += G 
			stateCnt[state] += 1
			G -= rewards[idx][index]
			G /= mdp.gamma

	## average calculation
	for state in mdp.states():
		if stateCnt[state] > epsilon:
			V[state] /= stateCnt[state]

	return V

####################################################################
## Monte Carlo Reinforcement Learning
## online learning based on new iteration

def epsilonGreedy(mdp, Q, state, epsilon = 1e-2):
	'''
	epsilon greedy policy function

	Args:
	mdp (class object):
	Q (dict): Q -func with key (state, action)
	state 
	epsilon (float)

	Return:
	action
	'''
	legalActions = [a[1] for a in mdp.getActions(state)]
	

	bestAction = max([(Q[(state, action)], action) for action in legalActions])[1]

	A = len(legalActions)
	probRange = []
	cnt = 0
	for i in range(A):

		if legalActions[i] != bestAction:
			base = epsilon / A
			pRange = (cnt * base, (cnt + 1) * base)
			cnt += 1
			probRange.append([legalActions[i], pRange])

	randomNum = random.random()

	action = None
	if randomNum >= epsilon - epsilon / A:
		action = bestAction

	else:
		for (act, pr) in probRange:
			mi, ma = pr 
			if randomNum >= mi and randomNum < ma:
				action = act 

	if action is None:
		print(probRange)
		raise Exception('No valid action!')

	return action

def computeError():
	'''
	Compute error
	'''
	pass 

def monteCarloSimulation(mdp, maxIter):
	'''
	Monte Carlo Simulation main function
	
	Args:
	maxIter (int): maximum iteration number 
	mdp (class object)

	Return:
	pi (dict): best policies 
	'''

	errorList = []
	Q = {}
	occurCnt = {}
	## initialize
	for state in mdp.states():
		legalActions = [a[1] for a in mdp.getActions(state)]
		for action in legalActions:
			Q[(state, action)] = 0
			occurCnt[state] = 0.001

	## random policies
	for it in range(maxIter):
		# optional compute error
		# errorList.append(computeError(Q))

		# simulate
		states, actions, rewards = simulate(mdp, Q)

		G = 0. 
		## accumulate reward
		for i in range(len(states)-1, -1, -1):
			G *= mdp.gamma
			G += rewards[i]
		## forward 
		for i in range(len(states)):
			## update Q
			state, action = states[i], actions[i]
			occurCnt[state] += 1.

			Q[(state, action)] = (Q[(state, action)] * \
							(occurCnt[state] - 1) + G) / occurCnt[state]


			G -= rewards[i]
			G /= mdp.gamma

	
	## get policies 
	pi = {}
	for state in mdp.states():
		if mdp.isEnd(state):
			pi[state] = None
			continue
		legalActions = [a[1] for a in mdp.getActions(state)]
		bestAction = max([(Q[(state, action)], action) for action in legalActions])[1]
		pi[state] = bestAction
	return pi


def simulate(mdp, Q, eg = False, stopStep = 100):
	'''
	simulate onces

	Args:
	mdp (class object): markov decision process
	Q (dict): Q function
	stopStep (int): in order to faster simulate, 
				we can set max stop step

	Return:
	states (list)
	actions (list)
	rewards (list)
	'''
	states = []
	actions = []
	rewards = []

	state = mdp.states()[int(random.random() * len(mdp.states()))]
	cnt = 0
	while not mdp.isEnd(state) and cnt <= stopStep:
		if eg:
			action = epsilonGreedy(mdp, Q, state)
		else:
			legalActions = [a[1] for a in mdp.getActions(state)]
			action = legalActions[int(random.random() * len(legalActions))]

		newState, reward = mdp.succAndReward(state, action)
		states.append(state)
		actions.append(action)
		rewards.append(reward)

		state = newState
		cnt += 1

	return states, actions, rewards

def getUniformSample(mu, var, sampleSize):
	'''
	get random sample that follows uniform distribution

	Args:
	mu (float): mean 
	var (float): variance
	sampleSize (int): how many samples to use

	Return:
	sample (list)
	'''
	sample = []
	for _ in range(sampleSize):
		ele = random.gauss(mu, var)
		sample.append(ele)
	return sample

if __name__ == '__main__':
	mdp = MDP(9)
	mdp = mdp_dp_solver.MazeMDP(5)
	pi = monteCarloSimulation(mdp, maxIter = 10000)
	state = mdp.startState()
	print(pi)
	while not mdp.isEnd(state):
		action = pi[state]
		newState, _ = mdp.succAndReward(state, action)
		print('State {} -> New State {} by Action {}'.format(state, newState, action))
		state = newState










