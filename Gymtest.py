# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:23:27 2019

@author: espeneil
"""

import gym
import random
import numpy as np
import tensorflow as tf
import SeeminglyWorking as dq
from collections import deque
import matplotlib.pyplot as plt

# matplotlib inline
print("Gym:", gym.__version__)
print("Tensorflow:", tf.__version__)

env_name = "CartPole-v0"
env = gym.make(env_name)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)


class QNetwork():
    def __init__(self, state_dim, action_size, tau=0.01):
        tf.reset_default_graph()
        self.state_in = tf.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(tf.float32, shape=[None])
        self.importance_in = tf.placeholder(tf.float32, shape=[None])
        action_one_hot = tf.one_hot(self.action_in, depth=action_size)

        self.q_state_local = self.build_model(action_size, "local")
        self.q_state_target = self.build_model(action_size, "target")

        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state_local, action_one_hot), axis=1)
        self.error = self.q_state_action - self.q_target_in
        self.loss = tf.reduce_mean(tf.multiply(tf.square(self.error), self.importance_in))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

        self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="local")
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")
        self.updater = tf.group([tf.assign(t, t + tau * (l - t)) for t, l in zip(self.target_vars, self.local_vars)])

    def build_model(self, action_size, scope):
        with tf.variable_scope(scope):
            hidden1 = tf.layers.dense(self.state_in, 64, activation=tf.nn.relu)
            hidden2 = tf.layers.dense(hidden1, 128, activation=tf.nn.relu)
            hidden3 = tf.layers.dense(hidden2, 64, activation=tf.nn.tanh)
            q_state = tf.layers.dense(hidden3, action_size, activation=None)
            return q_state

    def update_model(self, session, state, action, q_target, importance):
        feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target,
                self.importance_in: importance}
        error, _, _ = session.run([self.error, self.optimizer, self.updater], feed_dict=feed)
        return error

    def get_q_state(self, session, state, use_target=False):
        q_state_op = self.q_state_target if use_target else self.q_state_local
        q_state = session.run(q_state_op, feed_dict={self.state_in: state})
        return q_state

class PrioritizedReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1 / len(self.buffer) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return map(list, zip(*samples)), importance, sample_indices

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset


class DoubleDQNAgent():
    def __init__(self, state_size, action_size):
        self.state_dim = state_size#env.observation_space.shape
        self.action_size = action_size#env.action_space.n
        self.q_network = QNetwork(self.state_dim, self.action_size)
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=100000)
        self.gamma = 0.97
        self.eps = 1.0

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        action_greedy = np.argmax(q_state)
        action_random = np.random.randint(self.action_size)
        action = action_random if random.random() < self.eps else action_greedy
        return action

    def train(self, state, action, next_state, reward, done, use_DDQN=True, a=0.0):
        self.replay_buffer.add((state, action, next_state, reward, done))
        (states, actions, next_states, rewards, dones), importance, indices = self.replay_buffer.sample(120,priority_scale=a)
        next_actions = np.argmax(self.q_network.get_q_state(self.sess, next_states, use_target=False), axis=1)
        q_next_states = self.q_network.get_q_state(self.sess, next_states, use_target=use_DDQN)
        q_next_states[dones] = np.zeros([self.action_size])
        q_next_states_next_actions = q_next_states[np.arange(next_actions.shape[0]), next_actions]
        q_targets = rewards + self.gamma * q_next_states_next_actions
        errors = self.q_network.update_model(self.sess, states, actions, q_targets, importance ** (1 - self.eps))
        self.replay_buffer.set_priorities(indices, errors)

        if done: self.eps = max(0.1, 0.99 * self.eps)

    def __del__(self):
        self.sess.close()

#num_runs = 5
#run_rewards = []

#state_size = env.observation_space.shape
#action_size = env.action_space.n

#dagent = dq.DoubleDQNAgent(state_size, action_size)
'''
for n in range(num_runs):
    print("Run {}".format(n))
    ep_rewards = []
    agent = None
    agent = DoubleDQNAgent(state_size, action_size)# dq.DoubleDQNAgent(state_size, action_size)
    num_episodes = 200

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.get_action(state)
            print(action, 'action')
            next_state, reward, done, info = env.step(action)
            agent.train(state, action, next_state, reward, done, a=(n % 2 == 0) * 0.7)
            # env.render()
            total_reward += reward
            state = next_state

        ep_rewards.append(total_reward)
        # print("Episode: {}, total_reward: {:.2f}".format(ep, total_reward))

    run_rewards.append(ep_rewards)

for n, ep_rewards in enumerate(run_rewards):
    x = range(len(ep_rewards))
    cumsum = np.cumsum(ep_rewards)
    avgs = [cumsum[ep] / (ep + 1) if ep < 100 else (cumsum[ep] - cumsum[ep - 100]) / 100 for ep in x]
    col = "r" if (n % 2 == 0) else "b"
    plt.plot(x, avgs, color=col, label=n)


plt.title("Prioritized Replay performance")
plt.xlabel("Episode")
plt.ylabel("Last 100 episode average rewards")
plt.legend()
plt.show()

env.close()
'''