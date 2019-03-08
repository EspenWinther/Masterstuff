import numpy as np
import random
import matplotlib.pyplot as plt
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
        self.move = 0
        self.num_actions = len(self.action)
        self.onland = 0 #sm.sims[-1].val('LandEstimation','OnLand')
        self.terminal = 0
        self.nxtpos = (0.0,0.0) #next position
        self.nxtpossat = []
        self.pos = (0.0,0.0) #current position
        self.chosen_pos = (0.0,0.0)
        self.prevpos = (0.0,0.0)
        self.fakepos = (2.0,2.0)
        self.ori = () #current orientation
        self.reward = 0 #reward after finishing a move
        self.model = []
        self.states = []
        self.potential_waypoints = []
        self.x_pos = 0
        self.y_pos = 0
        self.number = 200 #changed from 50 #changed from 20
        self.R = 100
        self.goalReached = 0

    def get_reward(self):
        #Check where the vessel is and get the reward for that state
        print('got reward', self.reward)
        if self.pos in goal.goal_states:
            self.terminal = 1
            self.reward = goal.goal_reward
        elif self.terminal == 1:
            self.reward = neg_state.negativ_reward
        else:
            self.reward = ocean.ocean_reward

    def north(self):
        #Enable agent to go forward
        if self.chosen_pos[0] != 2000:
            self.nxtpos = (self.chosen_pos[0] + self.number, self.chosen_pos[1])
            self.fakepos = (self.chosen_pos[0] + self.number*6, self.chosen_pos[1])
#            self.get_reward()
        else:
            self.nxtpos = self.chosen_pos
#            self.get_reward()


    def south(self):
        #Enable agent to go backward
        if self.chosen_pos[0] != -2000:
            self.nxtpos = (self.chosen_pos[0] - self.number, self.chosen_pos[1])
            self.fakepos = (self.chosen_pos[0] - self.number*6, self.chosen_pos[1])
#            self.get_reward()
        else:
            self.nxtpos = self.chosen_pos
#            self.get_reward()

    def west(self):
        #Eable agent to turn left
        if self.chosen_pos[1] != 2000:
            self.nxtpos = (self.chosen_pos[0], self.chosen_pos[1] + self.number)
            self.fakepos = (self.chosen_pos[0], self.chosen_pos[1] + self.number*6)
#            self.get_reward()
        else:
            self.nxtpos = self.chosen_pos
#            self.get_reward()

    def east(self):
        #enable agent to turn right
        if self.chosen_pos[1] != -2000:
            self.nxtpos = (self.chosen_pos[0], self.chosen_pos[1] - self.number)
            self.fakepos = (self.chosen_pos[0], self.chosen_pos[1] - self.number*6)
#            self.get_reward()
        else:
            self.nxtpos = self.chosen_pos
#            self.get_reward()

    def north_west(self):
        if self.chosen_pos[0] == 2000 or self.chosen_pos[1] == 2000:
#            self.get_reward()
            self.nxtpos = self.chosen_pos
            
        else:
            self.nxtpos = (self.chosen_pos[0] + self.number, self.chosen_pos[1] + self.number)
            self.fakepos = (self.chosen_pos[0] + self.number*6, self.chosen_pos[1] + self.number*6)
#            self.get_reward()


    def north_east(self):
        if self.chosen_pos[0] == 2000 or self.chosen_pos[1] == -2000:
#            self.get_reward()
            self.nxtpos = self.chosen_pos
        else:
            self.nxtpos = (self.chosen_pos[0] + self.number, self.chosen_pos[1] - self.number)
            self.fakepos = (self.chosen_pos[0] + self.number*6, self.chosen_pos[1] - self.number*6)
#            self.get_reward()


    def south_west(self):
        if self.chosen_pos[0] == -2000 or self.chosen_pos[1] == 2000:
#            self.get_reward()
            self.nxtpos = self.chosen_pos
        else:
            self.nxtpos = (self.chosen_pos[0] - self.number, self.chosen_pos[1] + self.number)
            self.fakepos = (self.chosen_pos[0] - self.number*6, self.chosen_pos[1] + self.number*6)
#            self.get_reward()


    def south_east(self):
        if self.chosen_pos[0] == -2000 or self.chosen_pos[1] == -2000:
#            self.get_reward()
            self.nxtpos = self.chosen_pos
        else:
            self.nxtpos = (self.chosen_pos[0] - self.number, self.chosen_pos[1] - self.number)
            self.fakepos = (self.chosen_pos[0] - self.number*6, self.chosen_pos[1] - self.number*6)
#            self.get_reward()


    def wait(self):
        #Enable agent to wait
        self.nxtpos = self.chosen_pos
#        self.get_reward()
        
    def checkPos(self):
        self.prevpos = self.chosen_pos
        self.Cirlce = (self.nxtpos[0] - self.pos[0])**2 + (self.nxtpos[1] - self.pos[1])**2
        if self.Cirlce <= self.R**2:
            self.chosen_pos = self.nxtpos


class Goal_state():
    def __init__(self):
        #define init for goal state
        #Positive and terminal
        self.goal_reward = 10.
        self.goal_states = []

class Negative_state():
    def __init__(self):
        #define init for negative state
        #Negative and terminal
        self.negativ_reward = -10.
        self.negative_states = []

class Danger_state():
    def __init__(self):
        #Define init for dangerous state
        #Negative and non-terminal
        self.danger_reward = -5.
        self.danger_states = []

class Ocean():
    def __init__(self):
        #Define init for dangerous state
        #Negative and non-terminal
        self.ocean_reward = -1.
        self.ocean = []
        
agent = Agent()
neg_state = Negative_state()
ocean = Ocean()
danger = Danger_state()
goal = Goal_state()


def create_states():
    step = agent.number # forandret fra 50   #forandret fra 20
    stepx = -1
    stepy = -1
#    agent.Q_table[(0,0)] = [0.0]*agent.num_actions
#    agent.Q_table[(agent.startpos)] = [0.0]*agent.num_actions
    minarea = -2400
    maxarea = 2000
    for x in range(minarea,maxarea):
        stepx += 1
        if stepx == step:
            stepx = 0
            for y in range(minarea,maxarea):
                stepy +=1
                if stepy == step:
                    stepy = 0
                    agent.Q_table[(float(x+step),float(y+step))] = [0.0]*agent.num_actions
                    agent.potential_waypoints.append((x+step,y+step))



                        

def random_move():
#    move = 0
    chance = np.random.randint(0, 8)
    if chance == 0:
        agent.move = 0
        agent.west()
    elif chance == 1:
        agent.move = 1
        agent.east()
    elif chance == 2:
        agent.move = 2
        agent.north()
    elif chance == 3:
        agent.move = 3
        agent.south()
    elif chance == 4:
        agent.move = 4
        agent.north_west()
    elif chance == 5:
        agent.move = 5
        agent.north_east()
    elif chance == 6:
        agent.move = 6
        agent.south_west()
    elif chance == 7:
        agent.move = 7
        agent.south_east()
    elif chance == 8:
        agent.move = 8
        agent.wait()
#    return move



def best_move():
#    move = 0
    if agent.pos in agent.potential_waypoints:
        if max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][0]:
            agent.move = 0
            agent.west()
        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][1]:
            agent.move = 1
            agent.east()
        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][2]:
            agent.move = 2
            agent.north()
        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][3]:
            agent.move = 3
            agent.south()
        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][4]:
            agent.move = 4
            agent.north_west()
        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][5]:
            agent.move = 5
            agent.north_east()
        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][6]:
            agent.move = 6
            agent.south_west()
        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][7]:
            agent.move = 7
            agent.south_east()
        elif max(agent.Q_table[agent.chosen_pos]) == agent.Q_table[agent.chosen_pos][8]:
            agent.move = 8
            agent.wait()
#        return move


def bestVSrandom(epsilon):
    #make the best move or a random move
#    move = 0
    if random.uniform(0, 1) <= epsilon or sum(agent.Q_table[agent.chosen_pos]) == 0 :
        random_move()
    else:
        best_move()


def update_Q(alpha, gamma):
#    move = 0
    agent.get_reward()
    prev_pos = agent.prevpos
#    bestVSrandom(0.1)
    agent.states.append(prev_pos)
#    print('move',agent.move)
#    print('prev_pos',prev_pos)
#    print('chosen pos',agent.chosen_pos)
    if agent.terminal == 1:
        agent.Q_table[prev_pos][agent.move] = (1 - alpha) * agent.Q_table[prev_pos][agent.move] + alpha * agent.reward
    else:
        agent.Q_table[prev_pos][agent.move] = (1 - alpha) * agent.Q_table[prev_pos][agent.move] + alpha * (agent.reward + gamma * max(agent.Q_table[agent.chosen_pos]))
    agent.model.append(((prev_pos), (agent.chosen_pos), agent.move, agent.reward))
    # this part is the Dyna Q part
    print('before dyna',agent.Q_table[prev_pos][:])
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
        print('after dyna',agent.Q_table[state][action])
    return agent.Q_table



        
    