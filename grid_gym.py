#!/usr/bin/python 3.5
# -*-coding:utf-8:-*-

'''
build a world in the book
Author: Jing Wang
'''

import numpy as np 
import gym
from gym import spaces

class GridEnv(gym.Env):

	def render(self, mode = 'h'):
		from gym.envs.classic_control import rendering
		screen_width = 600
		screen_height = 600
		self.viewer = rendering.Viewer(screen_width, screen_height)

		#################################

		## create grid world 
		lines = []
		### horizontals
		for y in range(100, 580, 80):
			x1, x2 = 100, 500 
			li = rendering.Line((x1, y), (x2, y))
			lines.append(li)
		### verticals 
		for x in range(100, 580, 80):
			y1, y2 = 100, 500 
			li = rendering.Line((x, y1), (x, y2))
			lines.append(li)


		### create black squares
		vertices1 = [(100, 260), (100, 340), (180, 340), (180, 260)]
		vertices2 = [(260, 260), (260, 340), (180, 340), (180, 260)]
		vertices3 = [(260, 180), (260, 100), (340, 100), (340, 180)]
		vertices4 = [(420, 180), (420, 100), (340, 100), (340, 180)]
		vertices5 = [(420, 180), (420, 100), (500, 100), (500, 180)]
		vertices6 = [(340, 420), (420, 420), (420, 340), (340, 340)]
		vertices7 = [(340, 420), (420, 420), (420, 500), (340, 500)]
		vertices = [vertices1, vertices2, vertices3, vertices4, 
					vertices5, vertices6, vertices7]
		filledSquares = []
		for v in vertices:
			square = rendering.make_polygon(v, filled = True)
			square.set_color(0, 0, 0)
			filledSquares.append(square)


		## create exit
		exit = rendering.make_circle(40)
		trans = rendering.Transform(translation = (460, 300))
		exit.add_attr(trans)
		exit.set_color(255, 0, 0)

		## add objects
		for line in lines:
			self.viewer.add_geom(line)
		for square in filledSquares:
			self.viewer.add_geom(square)
		self.viewer.add_geom(exit)

		return self.viewer.render(return_rgb_array = mode == 'rgb_array')

