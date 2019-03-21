# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:21:08 2019

@author: Espen Eilertsen
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:36:28 2019

@author: Espen Eilertsen
https://www.youtube.com/watch?v=OYhFoMySoVs
based on kode by Keo Kim
"""

import random
import numpy as np
#import scenmanDqtest as sc
import tensorflow as tf
#import gym
from collections import deque #  deque is a list that can be added to in boths ends
from keras.models import Sequential
from keras.models import load_model
from keras.models import save_model
from keras.layers import Dense
from keras.optimizers import Adam
import os

'''
sett parameters
'''
#env = gym.make('CartPole-v0')
state_size = 5 #env.observation_space.shape[0] #  will output 4, number of "information" in the state space. [cartpos, cartspeed, armAngle, armspeed]
#print(state_size)
action_size = 4 #env.action_space.n #  will output 4, number of possible actions
#print(action_size)
batch_size = 64 #   can vary this by the power of two

n_episodes = 1000#  number of "games" we want the agent to play. more games = more data. randomly remember something from each episode as learning data

output_dir = "Weights_save/DqnWeights" #model_output/DQN_weights'
output_model = 'Model_mightWork.h5'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
'''
defining agent
'''
class DAgent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.state_size = state_size
        
        self.memory = deque(maxlen = 200000) #    creating memory, drops oldest after 2000. Only 2000 newest are interesting

        self.gamma = 0.96
        self.epsilon = 1.
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.learning_rate = 0.001
        self.model = self._build_model()
        
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(600, input_dim= self.state_size, activation='relu')) # first layer. [neurons, input_size, ]
        model.add(Dense(400, activation = 'relu'))
        model.add(Dense(300, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'softmax')) #change 'linear' to 'softmax'
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))

        return model
    

    def remember(self, state, action, reward, next_state, done):
       self.memory.append((state, action, reward, next_state, done))
       
    def act(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax (act_values[0])

    
    def replay(self,batch_size):
        minibatch = random.sample(self.memory,batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward #   if done = True
            if not done:
                variable = self.model.predict(next_state)[0]
                #print(np.argmax(variable))
                #print(reward)
                #print(self.gamma)
                target = reward + self.gamma*np.argmax(variable)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs = 1, verbose = 0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon * self.epsilon_decay
            
    def load(self, name):
        load_weights(name)

    def load_mod(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save_weights(name)

    def save_model(self, name):
        self.model.save(name)

dagent = DAgent(state_size, action_size)

'''
interact with environment
here is the real meat of the matter. this is how we use the functions created above
'''

#done = False
#for e in range(n_episodes):
#    state = sc.env_reset() #     reset environment to the beginning. In my case, set to start pos and heading. The env_reset()
#    state = np.reshape(state, [1, state_size]) #    transpose states so they fit with the network defined. transposing form row to collum
#
#    for time in range(5000): #   itterate over timesteps. is the pole alive in more then 5000 timesteps, it is complete. not valide for me
##        env.render() #  renders the game in action. also not needed
#        action = agent.act(state) #     here we choose the next action to be taken.
#        next_state, reward, done = sc.doAction(action) #    here is where the action is done. In my case: the simulator runs and we read off the next state when the action is complete
#
#        reward = reward if not done else -10 #  negative reward of we die. done = hitting land in my case.
#
#        next_state =np.reshape(next_state, [1, state_size])
#
#        agent.remember(state, action, reward, next_state, done)
#
#        state = next_state
#        
#        if done:
#            print(e,n_episodes,time,agent.epsilon)
#            break
#        
#        if len(agent.memory) >batch_size: #     Training agent theta(weights for network)
#            agent.replay(batch_size)
#            
#        if e % 50 == 0:
#            agent.save(output_dir + 'weights_' + '{:04d})'.format(e) + 'hdf5')
