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
        self.action = [0,1,2,3]
        self.move = 0
        self.num_actions = len(self.action)
        self.onland = 0#sm.sims[-1].val('LandEstimation','OnLand')
        self.terminal = 0
        self.nxtpos = (0.0,0.0) #agent next, or next,  position
        self.nxtpossat = []
        self.pos = (0.0,0.0) #current position
        self.chosen_pos = (0.0,0.0) #agent previous pos
        self.fakepos = (2.0,2.0)
        self.ori = () #current orientation
        self.reward = 0 #reward after finishing a move
        self.model = []
        self.states = []
        self.potential_waypoints = []
        self.possible_heading = []
        
        self.x_pos = 0
        self.y_pos = 0
        self.number = 100 #changed from 50 #changed from 2
        
        self.heading = 0
        self.heading_pos = 0
        self.heading_d = 0
        self.prev_ori = 0
        self. heading_state = []
        self.headingStart = 0
        
        self.R = 100
        self.speed = 0
        self.startpos = (-1400., 1400.)
        self.prevpos = 0
        
        self.chance = 0
        self.posreached = 0


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
        self.speed= 0.
        if self.headingStart < 35:
            self.heading_pos = self.possible_heading[self.headingStart +1]
            self.headingStart += 1
        elif self.heading_pos == self.possible_heading[35]:
            self.heading_pos = self.heading_pos
#        self.get_reward()
            
    def turn_left(self):
        #make agent turn left
        self.speed= 0.
        if self.headingStart > 0:
            self.heading_pos = self.possible_heading[self.headingStart -1]
            self.headingStart -= 1
        elif self.heading_pos == self.possible_heading[0]:
            self.heading_pos = self.heading_pos
            
#        self.get_reward()
            
    def move_forward(self):
        #make agent move forward
        self.speed= 0.5

    def wait(self):
        #Enable agent to wait
        self.speed= 0.
        self.posreached = 1


    def checkPos(self):
        self.prevpos = self.chosen_pos
#        self.nxtpos = self.chosen_pos
        self.dist = 50
        north = (self.chosen_pos[0], self.chosen_pos[1] + self.number)
        south = (self.chosen_pos[0], self.chosen_pos[1] - self.number)
        east = (self.chosen_pos[0] - self.number, self.chosen_pos[1])
        west = (self.chosen_pos[0] + self.number, self.chosen_pos[1])
        northwest = (self.chosen_pos[0] + self.number, self.chosen_pos[1] + self.number)
        northeast = (self.chosen_pos[0] - self.number, self.chosen_pos[1] + self.number)
        southwest = (self.chosen_pos[0] + self.number, self.chosen_pos[1] - self.number)
        southeast = (self.chosen_pos[0] - self.number, self.chosen_pos[1] - self.number)
#        print('north',north)
        if self.chosen_pos[0] <= -2000. or self.chosen_pos[0] >= 2000.:
            self.terminal = 1
            return 1
        elif self.chosen_pos[1] <=-2000. or self.chosen_pos[1] >= 2000.:
            self.terminal = 1
            return 1
        elif self.pos[0] >= north[0] - self.dist and self.pos[0] < north[0] + self.dist and self.pos[1] >= north[1] - self.dist and self.pos[1] < north[1] + self.dist:
            self.chosen_pos = north
            self.posreached = 1
        elif self.pos[0] >= south[0] - self.dist and self.pos[0] < south[0] + self.dist and self.pos[1] >= south[1] - self.dist and self.pos[1] < south[1] + self.dist:
            self.chosen_pos = south
            self.posreached = 1
        elif self.pos[0] >= east[0] - self.dist and self.pos[0] < east[0] + self.dist and self.pos[1] >= east[1] - self.dist and self.pos[1] < east[1] + self.dist:
            self.chosen_pos = east
            self.posreached = 1
        elif self.pos[0] >= west[0] - self.dist and self.pos[0] < west[0] + self.dist and self.pos[1] >= west[1] - self.dist and self.pos[1] < west[1] + self.dist:
            self.chosen_pos = west
            self.posreached = 1
        elif self.pos[0] >= northwest[0] - self.dist and self.pos[0] < northwest[0] + self.dist and self.pos[1] >= northwest[1] - self.dist and self.pos[1] < northwest[1] + self.dist:
            self.chosen_pos = northwest
            self.posreached = 1
        elif self.pos[0] >= northeast[0] - self.dist and self.pos[0] < northeast[0] + self.dist and self.pos[1] >= northeast[1] - self.dist and self.pos[1] < northeast[1] + self.dist:
            self.chosen_pos = northeast
            self.posreached = 1
        elif self.pos[0] >= southwest[0] - self.dist and self.pos[0] < southwest[0] + self.dist and self.pos[1] >= southwest[1] - self.dist and self.pos[1]  < southwest[1] + self.dist:
            self.chosen_pos = southwest
            self.posreached = 1
        elif self.pos[0] >= southeast[0] - self.dist and self.pos[0] < southeast[0] + self.dist and self.pos[1] >= southeast[1] - self.dist and self.pos[1] < southeast[1] + self.dist:
            self.chosen_pos = southeast
            self.posreached = 1
        else:
            self.posreached = 0


    
    def checkHeading(self):
        self.prev_ori = self.heading_pos
        if self.heading_pos - 0.2 < self.heading < self.heading_pos + 0.2:
            self.heading_d = self.heading_pos


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


#def create_states():
#    headingstep = 0
#    heading_step = 10
#    for headings in range(360):
#        headingstep +=1
#        if headingstep == heading_step:
#            agent.heading_state.append(round(headings,2))
#            headingstep = 0

def create_states():
    print('CREATE STATES')
    possible_heading = []
    for headings in range(0, 360):
        rads = headings*np.pi/180
        possible_heading.append(round(rads,2))
    print(len(possible_heading))
    steph = 0
    for headings in possible_heading:
        steph +=1
        if steph == 10:
            steph = 0
            agent.possible_heading.append(headings)
            
#    print('possible heading',possible_heading)
    step = agent.number # forandret fra 50   #forandret fra 20
    stepx = -1
    stepy = -1
#    steph = 9
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
                    for heading in agent.possible_heading:
                        agent.Q_table[(float(x+step),float(y+step)), heading] = [0.0]*agent.num_actions
                        agent.potential_waypoints.append(((x+step,y+step), heading))
#                            print(heading)
    print(len(agent.possible_heading))
#    print(agent.Q_table)
                        

def random_move():
    chance = np.random.randint(0, 3)
    if chance == 0:
        agent.move = agent.action[0]
        agent.turn_left()
    elif chance == 1:
        agent.move = agent.action[1]
        agent.turn_right()
    elif chance == 2:
        agent.move = agent.action[2]
        agent.move_forward()
    elif chance == 3:
        agent.move = agent.action[3]
        agent.wait()
    agent.chance = chance
#    return move



def best_move():
    if agent.pos in agent.potential_waypoints:
        if max(agent.Q_table[agent.chosen_pos, agent.heading_pos]) == agent.Q_table[agent.chosen_pos, agent.heading_pos][0]:
            agent.move = agent.action[0]
            agent.turn_right()
        elif max(agent.Q_table[agent.chosen_pos, agent.heading_pos]) == agent.Q_table[agent.chosen_pos, agent.heading_pos][1]:
            agent.move = agent.action[1]
            agent.turn_left()
        elif max(agent.Q_table[agent.chosen_pos, agent.heading_pos]) == agent.Q_table[agent.chosen_pos, agent.heading_pos][2]:
            agent.move = agent.action[2]
            agent.move_forward()
        elif max(agent.Q_table[agent.chosen_pos, agent.heading_pos]) == agent.Q_table[agent.chosen_pos, agent.heading_pos][3]:
            agent.move = agent.action[3]
            agent.wait()
#        return move


def bestVSrandom(epsilon):
    #make the best move or a random move
    if random.uniform(0, 1) <= epsilon or sum(agent.Q_table[agent.chosen_pos, agent.heading_pos]) == 0 :
        random_move()
    else:
        best_move()
#    return move


def update_Q(alpha, gamma):
    prev_pos, prev_or= agent.prevpos, agent.prev_ori
    agent.get_reward()
    agent.states.append((prev_pos, prev_or))
    if agent.terminal == 1:
        agent.Q_table[prev_pos,prev_or][agent.move] = (1 - alpha) * agent.Q_table[prev_pos,prev_or][agent.move] + alpha * agent.reward
    else:
        agent.Q_table[prev_pos,prev_or][agent.move] = (1 - alpha) * agent.Q_table[prev_pos,prev_or][agent.move] + alpha * (agent.reward + gamma * max(agent.Q_table[agent.chosen_pos, agent.heading_pos]))
    agent.model.append(((prev_pos,prev_or), (agent.chosen_pos, agent.heading_pos), agent.move, agent.reward))
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



        
    