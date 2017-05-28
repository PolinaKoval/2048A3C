import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import time, random, threading
from pandas import DataFrame
from keras.models import *
from keras.layers import *
from keras import backend as K
from NNs import getNN
import argparse
from env2048.env2048 import Game2048

parser = argparse.ArgumentParser()
parser.add_argument('-w', action="store_true")  # load weights
parser.add_argument('--test', action="store_true")  # test mode
parser.add_argument('-p', action="store_true")  # print score
parser.add_argument("--time", "-t", type=int, default=36000)
parser.add_argument("--env_type", type=int, default=0)
parser.add_argument("--nn_type", type=int, default=0)
parser.add_argument('--mask', action="store_true")
args = parser.parse_args()

# -- constants
RUN_TIME = args.time
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.002
SAVE_INTERVAL = 30000
k = 1

GAMMA = 0.99

N_STEP_RETURN = 1
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.9
EPS_STOP = .15
EPS_STEPS = 75000

MIN_BATCH = 100
LEARNING_RATE = 5e-4

LOSS_V = .4  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient
POWER = 11
STAT_FILE = "experiments/nn{}_env{}_mask{}_Stat.csv".format(args.nn_type, args.env_type, args.mask)
WEIGHTS_FILE = "experiments/nn{}_env{}_mask{}_weights.h5".format(args.nn_type, args.env_type, args.mask)

# ---------
class Brain:
	train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
	lock_queue = threading.Lock()

	def __init__(self, nn=0, mask=False, load_weights=False):
		input_layer, last_layer, placeholder, make_input, NONE_STATE = getNN(nn, mask);
		self.NONE_STATE = NONE_STATE
		self.make_input = make_input
		self.session = tf.Session()
		K.set_session(self.session)
		K.manual_variable_initialization(True)

		self.model = self._build_model(input_layer, last_layer)
		self.graph = self._build_graph(self.model, placeholder)

		self.session.run(tf.global_variables_initializer())
		self.default_graph = tf.get_default_graph()

		if load_weights and os.path.exists(WEIGHTS_FILE):
			self.load(WEIGHTS_FILE)
			print("weights loaded")

		self.default_graph.finalize()  # avoid modifications

	def _build_model(self, input_layer, last_layer):
		out_actions = Dense(NUM_ACTIONS, activation='softmax')(last_layer)
		out_value = Dense(1, activation='linear')(last_layer)

		model = Model(inputs=[input_layer], outputs=[out_actions, out_value])
		model._make_predict_function()  # have to initialize before threading

		return model

	def _build_graph(self, model, placeholder):
		s_t = placeholder
		a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
		r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

		p, v = model(s_t)

		log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
		advantage = r_t - v

		loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
		loss_value = LOSS_V * tf.square(advantage)  # minimize value error
		entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1,
											   keep_dims=True)  # maximize entropy (regularization)

		loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

		optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
		minimize = optimizer.minimize(loss_total)

		return s_t, a_t, r_t, minimize

	def optimize(self):
		if len(self.train_queue[0]) < MIN_BATCH:
			time.sleep(0)  # yield
			return

		with self.lock_queue:
			if len(self.train_queue[0]) < MIN_BATCH:  # more thread could have passed without lock
				return  # we can't yield inside lock

			s, a, r, s_, s_mask = self.train_queue
			self.train_queue = [[], [], [], [], []]

		s = np.array(s)
		a = np.vstack(a)
		r = np.vstack(r)
		s_ = np.array(s_)
		s_mask = np.vstack(s_mask)

		if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

		v = self.predict_v(s_)
		r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

		s_t, a_t, r_t, minimize = self.graph
		self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})
		global frames, k, df;
		if frames >= SAVE_INTERVAL * k:
			k += 1
			print(datetime.now())
			print("Saved weights and statistics for {} frames".format(frames))
			df.to_csv(STAT_FILE, index=False)
			self.save(WEIGHTS_FILE)

	def train_push(self, s, a, r, s_):
		with self.lock_queue:
			self.train_queue[0].append(s)
			self.train_queue[1].append(a)
			self.train_queue[2].append(r)

			if s_ is None:
				self.train_queue[3].append(brain.NONE_STATE)
				self.train_queue[4].append(0.)
			else:
				self.train_queue[3].append(s_)
				self.train_queue[4].append(1.)

	def predict(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)
			return p, v

	def predict_p(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)
			return p

	def predict_v(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)
			return v

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)


# ---------
frames = 0


class Agent:
	def __init__(self, eps_start, eps_end, eps_steps):
		self.eps_start = eps_start
		self.eps_end = eps_end
		self.eps_steps = eps_steps

		self.memory = []  # used for n_step return
		self.R = 0.

	def getEpsilon(self):
		if (frames >= self.eps_steps):
			return self.eps_end
		else:
			return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

	def act(self, s):
		eps = self.getEpsilon()
		global frames;
		frames = frames + 1

		if random.random() < eps:
			return random.randint(0, NUM_ACTIONS - 1)

		else:
			s = np.array([s])
			p = brain.predict_p(s)[0]

			a = np.argmax(p)
			# a = np.random.choice(NUM_ACTIONS, p=p)

			return a

	def train(self, s, a, r, s_):
		def get_sample(memory, n):
			s, a, _, _ = memory[0]
			_, _, _, s_ = memory[n - 1]

			return s, a, self.R, s_

		a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
		a_cats[a] = 1

		self.memory.append((s, a_cats, r, s_))

		self.R = (self.R + r * GAMMA_N) / GAMMA

		if s_ is None:
			while len(self.memory) > 0:
				n = len(self.memory)
				s, a, r, s_ = get_sample(self.memory, n)
				brain.train_push(s, a, r, s_)

				self.R = (self.R - self.memory[0][2]) / GAMMA
				self.memory.pop(0)

			self.R = 0

		if len(self.memory) >= N_STEP_RETURN:
			s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
			brain.train_push(s, a, r, s_)

			self.R = self.R - self.memory[0][2]
			self.memory.pop(0)

		# possible edge case - if an episode ends in <N steps, the computation is incorrect


# ---------
scores = []
set_of_states = []
# df = pd.read_csv('convergence.csv')
df = pd.DataFrame(columns=['AverageScore', 'AverageValueFn'])

class Environment(threading.Thread):
	stop_signal = False

	def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS, env_type=0):
		threading.Thread.__init__(self)

		self.render = render
		self.env = Game2048(4, env_type)
		self.agent = Agent(eps_start, eps_end, eps_steps)

	def runEpisode(self):

		s = self.env.reset()
		if len(set_of_states) < 30 and not any(np.array_equal(x, s) for x in set_of_states):
			set_of_states.append(s)
		s = brain.make_input(s)
		R = 0
		moved = True
		while True:
			time.sleep(THREAD_DELAY)  # yield

			if self.render: self.env.render()

			a = self.agent.act(s)
			s_, r, done, moved = self.env.step(a)
			s_ = brain.make_input(s_)
			if done:  # terminal state
				s_ = None

			self.agent.train(s, a, r, s_)

			s = s_
			R += r
			if done or self.stop_signal:
				break

		scores.append(self.env.score)

		if len(scores) > 500 and len(set_of_states) >= 30:
			v_fn = brain.predict_v(np.array(map(brain.make_input, set_of_states)))
			df.loc[df.shape[0]] = [np.average(scores), np.max(v_fn)]
			scores.pop(0)

		if args.p:
			print("Total R:{}\n".format(self.env.score))

	def run(self):
		while not self.stop_signal:
			self.runEpisode()

	def stop(self):
		self.stop_signal = True


# ---------
class Optimizer(threading.Thread):
	stop_signal = False

	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		while not self.stop_signal:
			brain.optimize()

	def stop(self):
		self.stop_signal = True


# -- main
env_test = Environment(eps_start=0.01, eps_end=0.01, env_type=args.env_type)
NUM_STATE = env_test.env.observation_space
NUM_ACTIONS = env_test.env.action_space

brain = Brain(nn=args.nn_type, load_weights=args.w, mask=args.mask)  # brain is global in A3C

envs = [Environment(env_type=args.env_type) for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

if args.test:
	print('Test')
	env_test.run()
else:
	print("Start training")
	for o in opts:
		o.start()

	for e in envs:
		e.start()

	time.sleep(RUN_TIME)

	for e in envs:
		e.stop()
	for e in envs:
		e.join()

	for o in opts:
		o.stop()
	for o in opts:
		o.join()

	brain.save(WEIGHTS_FILE)
	print("Training is completed")
