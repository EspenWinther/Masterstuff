# -*- coding: utf-8 -*-


'''
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''

from digitwin import DigiTwin
from scenrun import ScenarioRun
import scenparareader as spreader
import threading
from log import log, forcelog
import random
import math as m
import numpy as np
import NewNeuralNet as dq
from keras.utils import plot_model
import pickle
import timeit
import matplotlib.pyplot as plt
import tensorflow as tf

class Agent():
    def __init__(self):
        self.reward = 0
        self.chosen_pos = []
        self.speed = 0
        self.pos = 0
        self.heading = 0
        self.heading_d = 0
        self.heading_pos = 0
        self.dist = 50
        self.goal = 0
        self.negativ_reward = 0
        self.number = 50
        self.onland = 0
        self.goal_state = (0., 0.)
        self.possible_heading = []
        self.headingStart = 0
        self.num_actions = 4
        self.states = {}
        self.magnitudes = []
        self.nxtpos = (0.0, 0.0)
        self.success = 0
        self.failure = 0

    def get_reward(self):
        # Check where the vessel is and get the reward for that state
        if self.goal == 1:
            self.success += 1
            return 100
        elif self.onland == 1:
            self.failure += 1
            return -20
        else:
            self.reward = -1
            return -1

    def turn_right(self):
        # make the agent turn right
        self.speed = 0.
        if self.headingStart < 35:
            self.heading_pos = self.possible_heading[self.headingStart + 1]
            self.headingStart += 1
        elif self.heading_pos == self.possible_heading[35]:
            self.headingStart = 0
            self.heading_pos = self.possible_heading[self.headingStart]

    def turn_left(self):
        # make agent turn left
        self.speed = 0.
        if self.headingStart > 0:
            self.heading_pos = self.possible_heading[self.headingStart - 1]
            self.headingStart -= 1
        elif self.heading_pos == self.possible_heading[0]:
            self.headingStart = 35
            self.heading_pos = self.possible_heading[self.headingStart]

    def forward(self):
        # make agent move forward
        self.speed = 0.5

    def wait(self):
        # Enable agent to wait
        self.speed = 0.

    def checkPos(self):
        #        self.onland = 0
        self.dist = 25
        north = (self.chosen_pos[0], self.chosen_pos[1] + self.number)
        south = (self.chosen_pos[0], self.chosen_pos[1] - self.number)
        east = (self.chosen_pos[0] - self.number, self.chosen_pos[1])
        west = (self.chosen_pos[0] + self.number, self.chosen_pos[1])
        northwest = (self.chosen_pos[0] + self.number, self.chosen_pos[1] + self.number)
        northeast = (self.chosen_pos[0] - self.number, self.chosen_pos[1] + self.number)
        southwest = (self.chosen_pos[0] + self.number, self.chosen_pos[1] - self.number)
        southeast = (self.chosen_pos[0] - self.number, self.chosen_pos[1] - self.number)
        circle = (self.goal_state[0] - self.pos[0]) ** 2 + (self.goal_state[1] - self.pos[1]) ** 2
        R = 50
        # Adding increased goal target area for easy location
        if circle <= R ** 2:
            self.chosen_pos = self.goal_state
            self.goal = 1
        # ends here. remove this, circle and R to go back
        elif self.pos[0] >= north[0] - self.dist and self.pos[0] < north[0] + self.dist and self.pos[1] >= north[
            1] - self.dist and self.pos[1] < north[1] + self.dist:
            self.chosen_pos = north
        #            self.onland = 0
        elif self.pos[0] >= south[0] - self.dist and self.pos[0] < south[0] + self.dist and self.pos[1] >= south[
            1] - self.dist and self.pos[1] < south[1] + self.dist:
            self.chosen_pos = south
        #            self.onland = 0
        elif self.pos[0] >= east[0] - self.dist and self.pos[0] < east[0] + self.dist and self.pos[1] >= east[
            1] - self.dist and self.pos[1] < east[1] + self.dist:
            self.chosen_pos = east
        #            self.onland = 0
        elif self.pos[0] >= west[0] - self.dist and self.pos[0] < west[0] + self.dist and self.pos[1] >= west[
            1] - self.dist and self.pos[1] < west[1] + self.dist:
            self.chosen_pos = west
        #            self.onland = 0
        elif self.pos[0] >= northwest[0] - self.dist and self.pos[0] < northwest[0] + self.dist and self.pos[1] >= \
                northwest[1] - self.dist and self.pos[1] < northwest[1] + self.dist:
            self.chosen_pos = northwest
        #            self.onland = 0
        elif self.pos[0] >= northeast[0] - self.dist and self.pos[0] < northeast[0] + self.dist and self.pos[1] >= \
                northeast[1] - self.dist and self.pos[1] < northeast[1] + self.dist:
            self.chosen_pos = northeast
        #            self.onland = 0
        elif self.pos[0] >= southwest[0] - self.dist and self.pos[0] < southwest[0] + self.dist and self.pos[1] >= \
                southwest[1] - self.dist and self.pos[1] < southwest[1] + self.dist:
            self.chosen_pos = southwest
        #            self.onland = 0
        elif self.pos[0] >= southeast[0] - self.dist and self.pos[0] < southeast[0] + self.dist and self.pos[1] >= \
                southeast[1] - self.dist and self.pos[1] < southeast[1] + self.dist:
            self.chosen_pos = southeast

        if self.chosen_pos[0] <= -450.:
            self.onland = 1
        elif self.chosen_pos[0] >= 300.:
            self.onland = 1
        elif self.chosen_pos[1] <= -100.:
            self.onland = 1
        elif self.chosen_pos[1] >= 850.:
            self.onland = 1

    #            self.onland = 0
    #        print(self.chosen_pos)

    def checkHeading(self):
        self.prev_ori = self.heading_pos
        if self.heading_pos - 0.2 < self.heading < self.heading_pos + 0.2:
            self.heading_d = self.heading_pos


agent = Agent()


def create_states():
    print('CREATE STATES')
    possible_heading = []
    for headings in range(0, 360):
        rads = headings * np.pi / 180
        possible_heading.append(round(rads, 2))
    print(len(possible_heading))
    steph = 0
    for headings in possible_heading:
        steph += 1
        if steph == 10:
            steph = 0
            agent.possible_heading.append(headings)

    agent.magnitudes = [.0, .1, .2, .3, .4, .5, .6, .7]


#    step = agent.number # forandret fra 50   #forandret fra 20
#    stepx = -1
#    stepy = -1
##    steph = 9
#    minarea = -2400
#    maxarea = 2000
#    for magnitude in agent.magnitudes:
#        for x in range(minarea,maxarea):
#            stepx += 1
#            if stepx == step:
#                stepx = 0
#                for y in range(minarea,maxarea):
#                    stepy +=1
#                    if stepy == step:
#                        stepy = 0
#                        for heading in agent.possible_heading:
#                            for angel in agent.possible_heading:
#                                agent.states[(float(x+step),float(y+step)), heading, (magnitude, angel)] = [0.0]*agent.num_actions
##                                agent.Q_table[(float(x+step),float(y+step)), heading, (magnitude, angel)] = [0.0]*agent.num_actions
##                                agent.potential_waypoints.append(((x+step,y+step), heading,(magnitude, angel)))


# True if need to run without starting/using the Cybersea sim
NON_CS_DEBUG = False
# True if Cybersea config should be loaded at startup or not if want to save time
LOAD_CS_CFG = False
# not in use
USER_DIR = "C:\\temp"
# ports btw python app and Cybersea sim app
WEB_SERVER_PORT_INITIAL = 8085
PYTHON_PORT_INITIAL = 25338

SCEN_PARA_FILE = "scenparafile.txt"
# run or episode counter
run_ix = 0


# makes a list of all possible combinations of the run parameter range values
def make_para_vals():
    pv = para_min[:]
    pvs = [pv[:]]
    pix = paras - 1
    while pix > -1:
        pv[pix] += para_step[pix]
        if pv[pix] > para_max[pix]:
            pv[pix] = para_min[pix]
            pix -= 1
        else:
            pvs.append(pv[:])
            pix = paras - 1

    return pvs


####
# ***** EDITABLE *****
# get the next run parameter combination for the coming run
# default from ix=0 to end ("lowest" to "highest" point)
# 'bwd' = from ix=end to 0
# 'rand', initially randomly shuffled and then ix=0 to end.
# new ways can be implemented depending on exploration needs
###
def get_next_para_val(mode='fwd'):
    if len(para_vals) == 0:
        return None

    if mode == 'fwd':
        old_para_vals.append(para_vals.pop(0))
    elif mode == 'bwd':
        old_para_vals.append(para_vals.pop(-1))
    elif mode == 'rand':
        if len(old_para_vals) == 0:
            random.shuffle(para_vals)
        old_para_vals.append(para_vals.pop(0))
    else:  # add new ways above !
        forcelog('get_next_para_val mode is unknown!!')
        return None

    return old_para_vals[-1]


'''
 My stuff from here on
'''


def reset(string1, string2):
    sims[0].val(string1, string2, 1.)
    sims[0].step(20)
    sims[0].val(string1, string2, 0.)
    sims[0].step(50)


def env_reset():
    print('Reset!', agent.onland, done, agent.goal)
    agent.onland = 0
    agent.done = 0
    agent.goal = 0
    north = 700
    east = -250
    agent.headingStart = random.randint(0, len(agent.possible_heading) - 1)
    print(agent.onland, agent.done, agent.goal)
    reset('Hull', 'StateResetOn')
    sims[0].val('Hull', 'PosNED[0]', north)  # starty) #Y north
    sims[0].val('Hull', 'PosNED[1]', east)  # startx) #X east
    sims[0].val('Hull', 'ManualStateYaw', agent.possible_heading[agent.headingStart])  # X east
    x_pos = list(sims[0].val('Hull', 'Eta[1]'))[1]
    y_pos = list(sims[0].val('Hull', 'Eta[0]'))[0]
    agent.pos = (x_pos, y_pos)
    agent.heading_pos = agent.possible_heading[agent.headingStart]
    agent.chosen_pos = (east, north)

    reset('Hull', 'StateResetOn')
    print('success:', agent.success, 'failure:', agent.failure)
    state = buildState(0., 0.)
    return state


def calc_velvector():
    surge = sims[0].val('Hull', 'SurgeSpeed')
    sway = sims[0].val('Hull', 'SwaySpeed')
    vector = np.sqrt(surge ** 2 + sway ** 2)
    vectorSat = round(vector, 1)
    #    print(surge, sway)
    angle = np.arctan2(sway, surge)
    if angle < 0:
        angle += np.pi
    angleSat = round(angle, 2)
    for n in range(len(agent.possible_heading)):
        if agent.possible_heading[n] + 0.2 > angleSat > agent.possible_heading[n] - 0.2:
            angleSat = agent.possible_heading[n]

    return vectorSat, angleSat


def mapStates(OldValue, OldMin, OldMax, NewMin, NewMax):
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue


def buildState(vector, angle):
    headingmax = len(agent.possible_heading) - 1
    x = mapStates(agent.chosen_pos[0], -1500, 100, -1, 1)
    y = mapStates(agent.chosen_pos[1], -100, 1500, -1, 1)
    heading = mapStates(agent.heading_pos, agent.possible_heading[0], agent.possible_heading[headingmax], -1, 1)
    vectormag = mapStates(vector, 0, 0.5, -1, 1)
    vectorangle = mapStates(angle, agent.possible_heading[0], agent.possible_heading[headingmax], -1, 1)
    return [x, y, heading, vectormag, vectorangle]


def doAction(action):
    vector, angle = calc_velvector()
    agent.nxtpos = agent.chosen_pos
    #    agent.onland = 0
    done = 0

    if action == 0:
        agent.turn_left()
        sims[0].val('manualControl', 'UManual', agent.speed)
        sims[0].val('manualControl', 'PsiManual', agent.heading_pos)
        while agent.heading_d != agent.heading_pos:
            agent.onland = sims[0].val('LandEstimation', 'OnLand')
            agent.checkPos()
            if agent.onland == 1:
                done = 1
                break
            elif agent.goal == 1:
                done = 1
                break
            else:
                done = 0
            x_pos = list(sims[0].val('Hull', 'Eta[1]'))[1]
            y_pos = list(sims[0].val('Hull', 'Eta[0]'))[0]
            heading = list(sims[0].val('Hull', 'Eta[5]'))[5]

            agent.pos = (x_pos, y_pos)
            agent.heading = heading
            sims[0].step(10)
            agent.checkHeading()
            vector, angle = calc_velvector()

    elif action == 1:
        agent.turn_right()
        sims[0].val('manualControl', 'UManual', agent.speed)
        sims[0].val('manualControl', 'PsiManual', agent.heading_pos)
        while agent.heading_d != agent.heading_pos:
            agent.onland = sims[0].val('LandEstimation', 'OnLand')
            agent.checkPos()
            if agent.onland == 1:
                done = 1
                break
            elif agent.goal == 1:
                done = 1
                break
            else:
                done = 0
            x_pos = list(sims[0].val('Hull', 'Eta[1]'))[1]
            y_pos = list(sims[0].val('Hull', 'Eta[0]'))[0]
            heading = list(sims[0].val('Hull', 'Eta[5]'))[5]

            agent.pos = (x_pos, y_pos)
            agent.heading = heading
            sims[0].step(10)
            agent.checkHeading()
            vector, angle = calc_velvector()

    elif action == 2:
        agent.forward()
        sims[0].val('manualControl', 'UManual', agent.speed)
        sims[0].val('manualControl', 'PsiManual', agent.heading_pos)
        while agent.chosen_pos == agent.nxtpos:
            vector, angle = calc_velvector()
            agent.onland = sims[0].val('LandEstimation', 'OnLand')
            x_pos = list(sims[0].val('Hull', 'Eta[1]'))[1]
            y_pos = list(sims[0].val('Hull', 'Eta[0]'))[0]
            heading = list(sims[0].val('Hull', 'Eta[5]'))[5]
            agent.pos = (x_pos, y_pos)
            agent.heading = heading
            agent.checkPos()
            sims[0].step(10)
            agent.onland = sims[0].val('LandEstimation', 'OnLand')
            agent.checkPos()
            if agent.onland == 1:
                done = 1
                break
            elif agent.goal == 1:
                done = 1
                break
            else:
                done = 0
        print('chosen pos 0!', agent.chosen_pos[0], 'chosen pos 1!', agent.chosen_pos[1], 'heading', agent.heading_pos,
              'Done', done)


    elif action == 3:
        next_state = agent.wait()
        sims[0].step(10)
        print('Wait')

    next_state = buildState(vector, angle)
    reward = agent.get_reward()
    agent.checkPos()
    return next_state, reward, done


create_states()

'''
My stuff stops, main starts
'''

# MAIN...
if __name__ == "__main__":

    if LOAD_CS_CFG:
        log('LOAD_CS_CFG mode OFF !!!')
    if NON_CS_DEBUG:
        log('NON_CS_DEBUG mode ON !!!')

    # Initialize vectors
    sims = []
    sim_initiated = []
    sim_semaphores = []
    para_name = []
    para_min = []
    para_val = []
    para_max = []
    para_step = []
    runs = []
    init_runs = []
    run_scores = []

    # Load parameters from file
    sp_reader = spreader.ScenParaReader(SCEN_PARA_FILE)
    scen_para = sp_reader.para

    # reading in the simulation init parametres
    sim_init = scen_para['sim_init']
    CS_CONFIG_PATH = sim_init['CS_CONFIG_PATH']
    CS_PATH = sim_init['CS_PATH']
    THREADING = sim_init['THREADING']

    MAXACTIVETARGETSHIPS = int(sim_init['MAXACTIVETARGETSHIPS'])
    MAXTARGETSHIPS = int(sim_init['MAXTARGETSHIPS'])
    MAXTARGETSHIPWPS = int(sim_init['MAXTARGETSHIPWPS'])
    NUM_SIMULATORS = int(sim_init['NUM_SIMULATORS'])
    SIM_STEPS = int(sim_init['SIM_STEPS'])
    SIM_STEP = int(sim_init['SIM_STEP'])
    INIT_SIM_STEPS = int(sim_init['INIT_SIM_STEPS'])

    # reading in the rest of the parameters
    scen_init = scen_para['scen_init']
    run_init = scen_para['run_init']
    para_name = scen_para['name']
    para_min = scen_para['min']
    para_max = scen_para['max']
    para_step = scen_para['step']

    # number of run parameters (not values)
    paras = len(para_name)

    para_vals = make_para_vals()
    # make room for the already runned runs/episode parameter values
    old_para_vals = []

    # Start up all simulators
    for sim_ix in range(NUM_SIMULATORS):
        web_port = WEB_SERVER_PORT_INITIAL + sim_ix
        python_port = PYTHON_PORT_INITIAL + sim_ix
        log("Open CS sim " + str(sim_ix) + " @ port=" + str(web_port) + " Python_port=" + str(python_port))
        sims.append(None)
        sim_initiated.append(False)
        if not NON_CS_DEBUG:
            sims[-1] = DigiTwin('Sim' + str(1 + sim_ix), LOAD_CS_CFG, CS_PATH, CS_CONFIG_PATH, USER_DIR, web_port,
                                python_port)
        sim_semaphores.append(threading.Semaphore())

    log("Connected to simulators and configuration loaded")

    # Initiate the scenario of all simulators (scen_init parameters)
    # if this should be done for each run/episode, place parameters in run_init
    if len(scen_init):
        for sim_ix in range(NUM_SIMULATORS):
            if THREADING:
                sim_semaphores[sim_ix].acquire()
                log("Locking sim" + str(sim_ix + 1) + "/" + str(NUM_SIMULATORS))

            run_args = {'sim_init': sim_init, 'run_init': scen_init}
            run_name = 'Sim' + str(sim_ix + 1) + ' Run0'
            init_runs.append(ScenarioRun(run_name, sims[sim_ix], **run_args))

            if THREADING:
                t = threading.Thread(target=init_runs[-1].run, args=[0, INIT_SIM_STEPS, SIM_STEP], \
                                     kwargs={'semaphore': sim_semaphores[sim_ix], 'noscore': True})
                t.daemon = True
                t.start()

            else:
                init_runs[-1].run(0, INIT_SIM_STEPS, SIM_STEP, noscore=True)

    # Loop through the runs
    sim_ix = 0
    sim = sims[sim_ix]

    episode = 1
    episodes = 20
    done = 0
    '''
    This is where the magic happens!!!!!!
    '''
    #    dq.dagent.load("weights_+")
    #    mem = pickle.load(open('memroyPluss.p', 'rb'))
    #    for mems in mem:
    #        dq.dagent.memory.append(mems)
    #    print('memory',len(dq.dagent.memory))
    num_episodes = 2000
    num_runs = 5
    run_rewards = []
    dagent = None
    dagent = dq.DoubleDQNAgent()
    for n in range(num_runs):
        ep_rewards = []
        for eps in range(num_episodes):

            state = env_reset()  # reset environment to the beginning. In my case, set to start pos and heading. The env_reset()
            state = np.reshape(state, [5])
            done = 0
            agent.onland = sims[0].val('LandEstimation', 'OnLand')
            total_rewards = 0

            while not done:
                action = dagent.get_action(state)  # here we choose the next action to be taken.
                print(action)
                next_state, reward, done = doAction(action)  # here is where the action is done. In my case: the simulator runs and we read off the next state when the action is complete

                dagent.train(state, action, next_state, reward, done, a=(n%2==0)*0.7)
                total_rewards += reward
                state = next_state

        if eps % 50 == 0:
            dq.dagent.save("weightsPER")
            pickle.dump(dq.dagent.memory, open('memroyPER.p', 'wb'))
            # dq.dagent.save_model(dq.output_model)

            

        print('reward', reward)
        ep_rewards.append(total_rewards)
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
#plot_model(dq.output_dir, to_file='model.png')

#    while episode <= episodes:


#        episode += 1
#        #read ship state
#        curpos = [sim.val('Hull','Eta')[0], sim.val('Hull','Sway')]
#        curhead = sim.val('Hull','Yaw')
#
#        #calc targetship init-pos, init-course, init_speed
#        course = m.pi/2.
#        speed = 5
#        pos = [curpos[0]+400,curpos[1]-400]
#
#        #write to modules
#        #sim.val('module','parameter',val)
#        sim.val('TargetShipModule','courseInit[0]',course)
#        sim.val('TargetShipModule','psiManual[0]',course)
#        sim.val('TargetShipModule','speedInit[0]',speed)
#        sim.val('TargetShipModule','uManual[0]',speed)
#        sim.val('TargetShipModule','posNEDInit',pos)
#
#        #write StateResetOn if needed
#        #sim.val('module'.'StateResetOn',1)
#        sim.val('TargetShipModule','StateResetSingleTargetShip[0]',1)
#        #reset modules if needed (at least 20 steps)
#        sim.step(30)
#
#        #release StateResetOn if used above
#        #sim.val('module'.'StateResetOn',0)
#        sim.val('TargetShipModule','StateResetSingleTargetShip[0]',0)
#
#        #control vessel
#        steps = 1
#        while True:
#            #new control action
#            speed = 5
#            course = 0
#
#            sim.val('manualControl','UManual',speed)
#            sim.val('manualControl','PsiManual',course)
#
#            #step simulator
#            #a single step of the sim is 10ms
#            sim.step(SIM_STEP)
#
#            #read states
#            #sim.val('module','output')
#
#            #calculate reward
#
#            #terminate if needed
#            #for now just stop after a while
#            steps +=1
#            if steps*SIM_STEP > 10000:
#                break

