import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from env2048.env2048 import Game2048
import numpy as np
import copy
import random
env = Game2048(4)


def play_game():
	env.reset()
	while True:
		true_grid = copy.deepcopy(env.grid.get_values())
		best_a = 0
		best_r = 0
		moved_a = []
		for a in range(4):
			s, r, done, moved = env.step(a)
			if r > best_r:
				best_a = a
				best_r = r
			if moved:
				moved_a.append(a)
			env.grid.set_values(true_grid)

		if best_a not in moved_a:
			best_a = random.sample(moved_a, 1)[0]

		state, reward, done, moved = env.step(best_a)

		if done:
			return env.score, np.max(env.grid.get_values())


scores = []
max_cell = {}
for i in range(10000):
	# print i
	score, cell = play_game();
	scores.append(score)
	if cell not in max_cell:
		max_cell[cell] = 1
	else:
		max_cell[cell] += 1

print np.average(scores), np.max(scores), np.min(scores), np.median(scores)
print max_cell