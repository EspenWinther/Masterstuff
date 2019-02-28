import numpy as np
import random
import matplotlib.pyplot as plt
#import Digitwin
import digitwin
import math as m
from log import log, forcelog
import scenman as sm

class Agent():
    def __init__(self):
        #Define init for agent class
        self.Q_table = {}
        self.fails = []
        self.goals = []
        self.movment_cost = -1
        self.action = [0,1,2,3,4,5,6,7,8]
        self.num_actions = len(self.action)
        self.onland = 0#sm.sims[-1].val('LandEstimation','OnLand')
        self.terminal = 0
        self.nxtpos = (0.0,0.0) #next position
        self.nxtpossat = []
        self.pos = (0.0,0.0) #current position
        self.chosen_pos = (0.0,0.0)
        self.fakepos = (2.0,2.0)
        self.ori = () #current orientation
        self.reward = 0 #reward after finishing a move
        self.model = []
        self.states = []
        self.potential_waypoints = []
        self.x_pos = 0
        self.y_pos = 0
        self.number = 50 #changed from 50 #changed from 20
        self.heading = 0
        self.heading_pos = 0
        self. heading_state = []
        self.R = 50
        self.startpos = (-1501., 1499.)
#    def get_nxtsat(self):
#        for n in range(5):
#            for m in range(5):
#                self.nxtpossat.append((self.nxtpos[0]+n,self.nxtpos[1]+m))
#                self.nxtpossat.append((self.nxtpos[0]-n,self.nxtpos[1]-m))

    def get_reward(self):
        #Check where the vessel is and get the reward for that state
        if self.pos in goal.goal_states:
            self.terminal = 1
            self.reward == goal.goal_reward
        elif self.onland == 1:
            self.terminal = 1
            self.reward == neg_state.negativ_reward
        #elif self.pos in danger.danger_states:
         #   self.reward == danger.danger_reward
        else:
            self.reward == ocean.ocean_reward

    def turn_right(self):
        #make the agent turn right
        digitwin.sims[0].val('manualControl', 'UManual',0)
        self.heading_pos = self.heading + 0.174532925
        digitwin.sims[0].step(10)  
            
    def turn_left(self):
        #make agent turn left
        digitwin.sims[0].val('manualControl', 'UManual',0)
        self.heading_pos = self.heading - 0.174532925
        digitwin.sims[0].step(10)
            
    def move_forward(self):
        #make agent move forward
        while self.chosen_pos != self.nxtpos:
            digitwin.sims[0].val('manualControl', 'UManual',0.5)
            digitwin.sims[0].step(10)
            self.checkPos()
        digitwin.sims[0].val('manualControl', 'UManual',0)
#
    def wait(self):
        #Enable agent to wait
        self.nxtpos = self.chosen_pos
        self.get_reward()
        
    def checkPos(self):
        self.Cirlce = (self.nxtpos[0] - self.pos[0])**2 + (self.nxtpos[1] - self.pos[1])**2
        if self.Cirlce <= self.R**2:
            self.chosen_pos = self.nxtpos
        
    def checkHeading(self):
        if self.heading - 5 < self.heading < self.heading + 5:
            self.heading = self.heading_pos


class Goal_state():
    def __init__(self):
        #define init for goal state
        #Positive and terminal
        self.goal_reward = 10
        self.goal_states = (0.0,0.0)

class Negative_state():
    def __init__(self):
        #define init for negative state
        #Negative and terminal
        self.negativ_reward = -10
        self.negative_states = []

class Danger_state():
    def __init__(self):
        #Define init for dangerous state
        #Negative and non-terminal
        self.danger_reward = -5
        self.danger_states = []

class Ocean():
    def __init__(self):
        #Define init for dangerous state
        #Negative and non-terminal
        self.ocean_reward = -1
        self.ocean = []
        
agent = Agent()
neg_state = Negative_state()
ocean = Ocean()
danger = Danger_state()
goal = Goal_state()


def create_states():
    headingstep = 0
    heading_step = 10
    for headings in range(360):
        headingstep +=1
        if headingstep == heading_step:
            agent.heading_state.append(headings)
            headingstep = 0

    step = agent.number # forandret fra 50   #forandret fra 20
    stepx = 0
    stepy = 0
    agent.Q_table[(0,0)] = [0.0]*agent.num_actions
    agent.Q_table[(agent.startpos)] = [0.0]*agent.num_actions
    minarea = -2000
    maxarea = 2000
    for x in range(minarea,maxarea):
        stepx += 1
        if stepx == step:
            stepx = 0
            for y in range(minarea,maxarea):
                stepy +=1
                if stepy == step:
                    stepy = 0
                    for heading_pos in agent.heading_state:
                        agent.Q_table[(float(x+step),float(y+step)),heading_pos] = [0.0]*agent.num_actions
                        agent.potential_waypoints.append((x+step,y+step),heading_pos)

                        

def random_move():
    chance = np.random.randint(0, 9)
    if chance == 0:
        move = agent.action[0]
        agent.turn_left()
    elif chance == 1:
        move = agent.action[1]
        agent.turn_right()
    elif chance == 2:
        move = agent.action[2]
        agent.move_forward()
    elif chance == 3:
        move = agent.action[3]
        agent.wait()
#    elif chance == 4:
#        move = agent.action[4]
#        agent.north_west()
#    elif chance == 5:
#        move = agent.action[5]
#        agent.north_east()
#    elif chance == 6:
#        move = agent.action[6]
#        agent.south_west()
#    elif chance == 7:
#        move = agent.action[7]
#        agent.south_east()
#    elif chance == 8:
#        move = agent.action[8]
#        agent.wait()
    return move



def best_move():
    if agent.pos in agent.potential_waypoints:
        if max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][0]:
            move = agent.action[0]
            agent.turn_right()
        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][1]:
            move = agent.action[1]
            agent.turn_left()
        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][2]:
            move = agent.action[2]
            agent.move_forward()
        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][3]:
            move = agent.action[3]
            agent.wait()
#        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][4]:
#            move = agent.action[4]
#            agent.north_west()
#        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][5]:
#            move = agent.action[5]
#            agent.north_east()
#        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][6]:
#            move = agent.action[6]
#            agent.south_west()
#        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][7]:
#            move = agent.action[7]
#            agent.south_east()
#        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][8]:
#            move = agent.action[8]
#            agent.wait()
        return move


def bestVSrandom(epsilon):
    #make the best move or a random move
    if random.uniform(0, 1) <= epsilon or sum(agent.Q_table[agent.chosen_pos]) == 0 :
        move = random_move()
    else:
        move = best_move()
    return move


def update_Q(alpha, gamma):
    prev_pos, prev_or, prew_reward = agent.chosen_pos, agent.ori, agent.reward
    move = bestVSrandom(0.1)
    agent.states.append(prev_pos)
    if agent.terminal == 1:
        agent.Q_table[prev_pos][move] = (1 - alpha) * agent.Q_table[prev_pos][move] + alpha * agent.reward
    else:
        agent.Q_table[prev_pos][move] = (1 - alpha) * agent.Q_table[prev_pos][move] + alpha * (agent.reward + gamma * max(agent.Q_table[agent.chosen_pos]))
    agent.model.append(((prev_pos), (agent.chosen_pos), move, agent.reward))
    # this part is the Dyna Q part
    if len(agent.model) >20:
        for number in range(100):
            x = np.random.randint(0,len(agent.model))
            state = agent.model[x][0]
            nextstate = agent.model[x][1]
            action = agent.model[x][2]
            reward = agent.model[x][3]
            agent.Q_table[state][action] = (1 - alpha) * agent.Q_table[state][action] + alpha * (reward + gamma * max(agent.Q_table[nextstate]))
            # print(reward)
            # print(action)
            # print(nextstate)
            # print(state)
    return agent.Q_table



        
    