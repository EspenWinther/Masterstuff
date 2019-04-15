import numpy as np
import tensorflow as tf
from collections import deque
import random

class Qnetwork():
    def __init__(self, action_size, tau = 0.01):
        tf.reset_default_graph()
        self.state_in = tf.placeholder(tf.float32, shape=[None,*(5,)])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(tf.float32, shape=[None])
        self.importance_in = tf.placeholder(tf.float32, shape=[None])
        action_one_hot = tf.one_hot(self.action_in, depth=action_size)

        self.q_state_local = self.build_model(action_size, "local")
        self.q_state_target = self.build_model(action_size, "target")

        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state_local, action_one_hot), axis = 1)
        self.error = self.q_state_action - self.q_target_in
        self.loss = tf.reduce_mean(tf.multiply(tf.square(self.error),self.importance_in))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

        self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="local")
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")
        self.updater = tf.group([tf.assign(t,t+tau*(1-t)) for t,l in zip(self.target_vars, self.local_vars)])


    def build_model(self, action_size, scope):
        with tf.variable_scope(scope):
            first = tf.layers.dense(self.state_in, 100, activation=tf.nn.tanh)
            dropout1 = tf.layers.dropout(first, 0.25)
            second = tf.layers.dense(dropout1, 300, activation=tf.nn.tanh)
            dropout2 = tf.layers.dropout(second, 0.25)
            third = tf.layers.dense(dropout2, 300, activation=tf.nn.tanh)
            dropout3 = tf.layers.dropout(third, 0.25)
            fourth = tf.layers.dense(dropout3, 100, activation=tf.nn.tanh)
            output = tf.layers.dense(fourth, action_size)
            return output

    def update_model(self, session, state, action, q_target, importance):
        feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target, self.importance_in: importance}
        error,_,_ = session.run([self.error, self.optimizer, self.updater], feed_dict = feed)
        return error

    def get_q_state(self, session, state, use_target=False):
        #print('5')
        q_state_op = self.q_state_target if use_target else self.q_state_local
        #print('6')
        q_state = session.run(q_state_op, feed_dict= {self.state_in: state})
        return q_state

class PriRepBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))

    def get_probs(self, priority_scale):
        scaled_priorities = np.array(self.priorities)**priority_scale
        sample_probabilities = scaled_priorities/sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1 / len(self.buffer) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale = 1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probs(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return map(list, zip(*samples)), importance, sample_indices

    def set_priorities(self, indices, errors, offset= 0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset


class DoubleDQNAgent():
    def __init__(self):
        self.state_size = 5
        self.action_size = 4
        self.q_network = Qnetwork(self.action_size)
        self.replay_buffer = PriRepBuffer(maxlen=100000)
        self.gamma = 0.97
        self.epsilon = 1.0
        self.onland = 0
        self.goal = 0

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        action_greedy = np.argmax(q_state)
        action_random = np.random.randint(self.action_size)
        action = action_random if random.random() < self.epsilon else action_greedy
        return action

    def train(self, state, action, next_state, reward, done, use_DDQN=True, a=0.0):
        self.replay_buffer.add((state, action, next_state, reward, done))
        #print(state, action, next_state, reward, done)
        (states, actions, next_states, rewards, dones), importance, indices = self.replay_buffer.sample(50, priority_scale = a)
        #print(states, actions, next_states, rewards, dones)
        next_actions = np.argmax(self.q_network.get_q_state(self.sess, next_states, use_target=False), axis = 1)
        #print('1')
        q_next_state = self.q_network.get_q_state(self.sess, next_states, use_target=True)
        #print('2')
        q_next_state[dones] = np.zeros([self.action_size])
        #print('3')
        q_next_state_next_actions = q_next_state[np.arange(next_actions.shape[0]), next_actions]
        #print('4')
        q_targets = reward + self.gamma*q_next_state_next_actions
        #print('before update model')
        errors = self.q_network.update_model(self.sess, states, actions, q_targets, importance**(1-self.epsilon))
        self.replay_buffer.set_priorities(indices, errors)

        if done and len(self.replay_buffer.buffer) > 3000:
            self.epsilon = max(0.1, 0.00099* self.epsilon)

    def __del__(self):
       self.sess.close()

