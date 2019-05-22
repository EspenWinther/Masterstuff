# -*- coding: utf-8 -*-
"""
Scenario Manager
Starting the simulator(s)
reading the file parameters
looping the scenario runs
if >1 simulators, threading should be used

"""
import digitwin
from digitwin import DigiTwin
from scenrun import ScenarioRun
import scenparareader as spreader
import threading
from log import log, forcelog
import random
import PathIdeaAgent as Rl
import numpy as np
import math
import os
import LOS
import pickle
# import SeeminglyWorking as dq#NewNeuralNetPath as dq
import DQN_demo as dqn
import SeeminglyWorking as se
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import Getscreen as snap
import TensorNet as tn
import Gymtest as gm

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

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

#output_dir = "convex_landNet/provenconv_network/funckyMoves"  # model_output/DQN_weights'
output_dir =  "convex_landNet/provenconv_network/differentMoves_5"
output_dir2 =  "convex_landNet/provenconv_network/differentMoves_enemy"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

class Enemy():
    def __init__(self):
        self.pos = [0,0]
        self.move = 0
        self.path = [(0,0), (50,0), (100,0),(150,0),(200,0),(250,0),(300,-50),(300,-100),(350,-150),(400,-150),(450,-150),(500,-150), (600,-150)]
        sims[0].val('TargetShipModule', 'activateWaypoints', 0.)


    def reset(self):
        self.pos = self.path[0]
        self.move = 0
        sims[0].val('TargetShipModule', 'posNEDInit[0]', self.pos[0])
        sims[0].val('TargetShipModule', 'posNEDInit[1]', self.pos[1])

    def action(self):
        if self.pos != self.path[len(self.path)-1]:
            self.move += 1
            self.pos = self.path[self.move]
        else:
            self.pos = self.path[len(self.path)-1]
        sims[0].val('TargetShipModule', 'posNEDInit[0]', self.pos[0])
        sims[0].val('TargetShipModule', 'posNEDInit[1]', self.pos[1])




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


##MAIN...
if __name__ == "__main__":

    if LOAD_CS_CFG:
        log('LOAD_CS_CFG mode OFF !!!')
    if NON_CS_DEBUG:
        log('NON_CS_DEBUG mode ON !!!')
    #    digitwin.DigiTwin.setRealTimeMode()

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
        print('sim_ix')
    log("Connected to simulators and configuration loaded")

    enemy = Enemy()

    def reset(string1, string2):
        sims[0].val(string1, string2, 1.)
        sims[0].step(20)
        sims[0].val(string1, string2, 0.)
        sims[0].step(50)


    def env_reset(transfer):
        print('terminal', Rl.agent.terminal)
        print('onland!')
        reset('Hull', 'StateResetOn')
        Y = 400
        X = -150
        X, Y = random_start(transfer)
        Rl.agent.chosen_pos = (X, Y)
        moveShip()
        onland = sims[0].val('LandEstimation', 'OnLand')
        while onland == 1:
            print('Retry random start')
            X, Y = random_start()
            Rl.agent.chosen_pos = (X, Y)
            moveShip()
            onland = sims[0].val('LandEstimation', 'OnLand')
        Rl.agent.onland = 0
        Rl.agent.goal = 0
        Rl.agent.heading = 0
        Rl.agent.moving = 0
        enemy.reset()
        reset('Hull', 'StateResetOn')
        state = buildStatePath()
        # state = snap.grab_pic() #1200x800 frame
        # state = [Rl.agent.chosen_pos[0], Rl.agent.chosen_pos[1]]
        return state


    def sat(value):
        num = math.ceil(value)
        return num


    def moveShip():
        reset('Hull', 'StateResetOn')
        sims[0].val('Hull', 'PosNED[0]', Rl.agent.chosen_pos[1])
        sims[0].val('Hull', 'PosNED[1]', Rl.agent.chosen_pos[0])
        reset('Hull', 'StateResetOn')


    def mapStates(OldValue, OldMin, OldMax, NewMin, NewMax):
        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        return NewValue


    land_mass = deque(maxlen=60)
    for x in range(60):
        land_mass.append((0, 0))



    def buildStatePath(minx=-300, miny=-50, maxx=150, maxy=500, R=50):
        myx = Rl.agent.chosen_pos[0]
        myy = Rl.agent.chosen_pos[1]
        goal_dist = np.sqrt(Rl.agent.chosen_pos[0] ** 2 + Rl.agent.chosen_pos[1] ** 2)
        max_dist = np.sqrt(maxx ** 2 + maxy ** 2)
        dist = mapStates(goal_dist, R, max_dist, 0, 5)
        trafficx = enemy.pos[0]#list(sims[0].val('TargetShipModule', 'PositionX[0]'))[1]
        trafficy = enemy.pos[1]#list(sims[0].val('TargetShipModule', 'PositionY[0]'))[1]
        trafx = mapStates(trafficx, minx, maxx, 0, 5)  # chosen_pos[0], -1500, 100, -1, 1)
        trafy = mapStates(trafficy, miny, maxy, 0, 5)
        myx = mapStates(myx, minx, maxx, 0, 5)  # chosen_pos[0], -1500, 100, -1, 1)
        myy = mapStates(myy, miny, maxy, 0, 5)  # chosen_pos[1], -100, 1500, -1, 1)
        heading = mapStates(Rl.agent.heading, 0, 7, 0, 5)  # chosen_pos[0], -1500, 100, -1, 1)
        moving = mapStates(Rl.agent.moving, 0, 1, 0, 5)  # chosen_pos[1], -100, 1500, -1, 1)
        pi = np.pi
        goal_angle = np.arctan2(Rl.agent.chosen_pos[0], Rl.agent.chosen_pos[1])
        angle = mapStates(goal_angle, -pi, pi, -1, 1)
        state = [myx, myy, heading, moving, dist, trafx, trafy]
        #for things in land_mass:
         #   things2 = np.sqrt((myx - things[0]) ** 2 + (myy - things[1]) ** 2)
          #  state.append(things2)
        return state


    def keeptrack():
        path = Rl.agent.pathulus
        R = 50
        for n in path:
            myx = list(sims[0].val('Hull', 'Eta[1]'))[1]
            myy = list(sims[0].val('Hull', 'Eta[0]'))[0]
            circle = (Rl.agent.path[n][0] - myx) ** 2 + (Rl.agent.path[n][1] - myy) ** 2
            if circle <= R ** 2:
                chosen_pos = (Rl.agent.path[n][0], Rl.agent.path[n][1])
        return chosen_pos


    def checkPos(e, transfer):
        # Rl.agent.chosen_pos = Rl.agent.nxtpos
        circle = (Rl.agent.goal_state[0] - Rl.agent.chosen_pos[0])**2 + (Rl.agent.goal_state[1] - Rl.agent.chosen_pos[1])**2
        traffic_circle = (enemy.pos[1] - Rl.agent.chosen_pos[0])**2 + (enemy.pos[0] - Rl.agent.chosen_pos[1])**2
        print('traffic', traffic_circle, 'enemy position', enemy.pos)
        R = 120
        r = 75
        if Rl.agent.onland == 0:
            Rl.agent.onland = sims[0].val('LandEstimation', 'OnLand')
        # Adding increased goal target area for easy location
        #if transfer == 1:
        if Rl.agent.chosen_pos[1] > 500:  # 1550:
            Rl.agent.onland = 1
        elif Rl.agent.chosen_pos[1] < -50:  # -50:
            Rl.agent.onland = 1
        elif Rl.agent.chosen_pos[0] > 150:  # 50:
            Rl.agent.onland = 1
        elif Rl.agent.chosen_pos[0] < -300:  # -1550:
            Rl.agent.onland = 1
        elif traffic_circle <= r**2:
            Rl.agent.onland = 1
            Rl.agent.crash += 1
            done = True
        #else:
        #    if Rl.agent.chosen_pos[1] > 300:  # 1550:
        #        Rl.agent.onland = 1
        #    elif Rl.agent.chosen_pos[1] < -50:  # -50:
        #        Rl.agent.onland = 1
        #    elif Rl.agent.chosen_pos[0] > 150:  # 50:
        #        Rl.agent.onland = 1
        #    elif Rl.agent.chosen_pos[0] < -100:  # -1550:
        #        Rl.agent.onland = 1

        if Rl.agent.onland == 1:
            done = True
            Rl.agent.consec_goals = 0
        elif circle <= R ** 2:
            Rl.agent.chosen_pos = Rl.agent.goal_state
            Rl.agent.goal = 1
            Rl.agent.consec_goals += 1
            done = False
            if Rl.agent.consec_goals >= 20:
                done = True
        else:
            done = False

        return done


    pathulus = []


    def get_reward(invalid=0):
        # Check where the vessel is and get the reward for that state
        #        self.onland = sims[0].val('LandEstimation','OnLand')
        if invalid == 1:
            return -2
        if Rl.agent.onland == 1:
            Rl.agent.failure += 1
            return -1  # was -1
        elif Rl.agent.goal == 1:
            Rl.agent.success += 1
            return 2  # was 2ew
        else:
            Rl.agent.reward = 0
            return -0.001


    def printAction():
        if Rl.agent.move == 0:
            print('GO')
        elif Rl.agent.move == 1:
            print('turning_right')
        elif Rl.agent.move == 2:
            print('turning_left')
        elif Rl.agent.move == 3:
            print('stop')
        elif Rl.agent.move == 4:
            print('Slide_left')
        elif Rl.agent.move == 5:
            print('Slide_right')
        elif Rl.agent.move == 6:
            print('waiting')


    def random_start(transfer=0):
        # x = [-1350,-1300,-1250,-1200,-1150,-1100,-1050,-1000,-950,-900,-850,-800]
        # y = [1350,1300,1250,1200,1150,1100,1050,1000,950,900,850,800,750,700,650]
        # x2 = [-300, -250, -200, -150, -100, -50, 0]
        # y2 = [600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100]
        #pos = [(400, -150), (300, -150), (200, -200)]
        pos = [(400, -150), (200, -200), (300, -150)]
        pos1 = [(300,0), (300,-50), (200,50)]
        pos2 = [(400, -150), (200, -200), (300, -150)]
        c_pos = pos[random.randrange(len(pos))]
        if transfer ==1:
            c_pos = pos[random.randrange(len(pos))]
        x_pos, y_pos = c_pos[1], c_pos[0]
        # xout = random.choice(x2)
        # yout = random.choice(y2)
        return x_pos, y_pos


    def Correct_input(pos):
        if pos == (300,0):
            action = []

    def plot_goals(inpt, goal, fail, crash, string, fig):
        # plt.plot(input, agent.crash, 'b', label='Crash')
        plt.figure(fig)
        plt.plot(inpt, goal, 'g', label='Goals')
        plt.plot(inpt, fail, 'r', label='Fails')
        plt.plot(inpt, crash, 'b', label='Crash')
        plt.legend(loc='upper left')
        plt.xlabel('Iterations')
        plt.ylabel('Fails/Goals/Crashes')
        plt.title('DQN')
        plt.grid(True)
        plt.savefig(string)
        # plt.show()


    def append_landmass(minx=-300, miny=-50, maxx=50, maxy=500, R=50):
        landx = mapStates(Rl.agent.chosen_pos[0], minx, maxx, -1, 1)  # chosen_pos[0], -1500, 100, -1, 1)
        landy = mapStates(Rl.agent.chosen_pos[1], miny, maxy, -1, 1)  # chosen_pos[1], -100, 1500, -1, 1)
        if (landx, landy) not in land_mass:
            land_mass.append((landx, landy))


    sim_ix = 0
    minx, miny, maxx, maxy = Rl.create_states()

    sims[0].val('LOSGuidance', 'numWaypoints', 0.)
    sims[0].val('LOSGuidance', 'activateWaypoints', 0.)

    running = 1
    num_episodes = 10000
    number = 0
    terminalNumber = 0
    Rl.agent.onland = sims[0].val('LandEstimation', 'OnLand')
    # Rl.agent.Q_table = pickle.load(open('progressPath.p', 'rb')) #loading the Qtable variable
    # dagent = dq.DoubleDQNAgent(3,8)

    iteration = []
    shape_state = [None, 100, 120, 1]
    other_shape = [7]
    seAgent = gm.DoubleDQNAgent(other_shape, 6)


    # dq.restore(dagent.sess, 'First_phase')
    # seAgent = tn.Agent(shape_state,8)
    amount_states = 7#len(land_mass) + 3  # [100, 120, 1]
    ep_rewards = []
    run_rewards = []
    total_rewards = 0
    eps_rewards = 0

    if THREADING:
        sim_semaphores[sim_ix].acquire()
        log("Locking sim" + str(sim_ix + 1) + "/" + str(NUM_SIMULATORS))
    #   state = [chosen_pos, heading, agent.moving, traffic]

    iteration = []
    p = 0
    fig = 1
    timeToSave = 0
    transfer = 0
    tf.train.Saver().restore(seAgent.sess, output_dir)
    episode_num = 0
    for n in range(10):  # runs
        #iteration.clear()
        #Rl.agent.goals.clear()
        #Rl.agent.fails.clear()
        #Rl.agent.crashes.clear()
        for e in range(1000):  # episodes
            episode_num += 1
            done = False
            total_reward = 0
            state = env_reset(transfer)  # reset environment to the beginning. In my case, set to start pos and heading. The env_reset()
            Rl.agent.onland = sims[0].val('LandEstimation', 'OnLand')
            sims[0].step(10)
            state = np.reshape(state, amount_states)  # transpose states so they fit with the network defined. transposing form row to collum
            while not done:
                print('episode', e, 'success', Rl.agent.success, 'failure', Rl.agent.failure)
                prev_pos = Rl.agent.chosen_pos
                action = seAgent.get_action(state)  # here we choose the next action to be taken.
                invalid = Rl.doAction(action)
                moveShip()
                enemy.action()
                done = checkPos(e, transfer)
                reward = get_reward()
                total_reward += reward
                print('reward', reward)
                next_state = buildStatePath()
                next_state = np.reshape(next_state, amount_states)
                seAgent.train(state, action, next_state, reward, done, a=.7)
                state = next_state
                #agent.train(state, action, next_state, reward, done, a=(n % 2 == 0) * 0.7)

                if Rl.agent.goal == 1:
                    state = env_reset(transfer)
                    state = np.reshape(state, amount_states)
                    if Rl.agent.consec_goals >= 20:
                        transfer = 1
                if done:
                    append_landmass()
                    print('done', e, n)
                    iteration.append(episode_num)
                    Rl.agent.goals.append(Rl.agent.success)
                    Rl.agent.fails.append(Rl.agent.failure)
                    Rl.agent.crashes.append(Rl.agent.crash)
                    # dagent.save("weights_pathonly_easy")
                    timeToSave+=1
                    if timeToSave == 50:
                        tf.train.Saver().save(seAgent.sess, output_dir2)
                        timeToSave = 0

                print('total reward', total_reward)
            ep_rewards.append(total_reward)
        run_rewards.append(ep_rewards)
        savegoals = "tenthfolder/Goal_differentMove_enemy11%i" % p
        savescore = "tenthfolder/Score_differentMove_enemy11%i" % p
        p += 1
        fig += 1
        plot_goals(iteration, Rl.agent.goals, Rl.agent.fails, Rl.agent.crashes, savegoals, fig)

        for n, ep_rewards in enumerate(run_rewards):
            x = range(len(ep_rewards))
            cumsum = np.cumsum(ep_rewards)
            avgs = [cumsum[ep] / (ep + 1) if ep < 100 else (cumsum[ep] - cumsum[ep - 100]) / 100 for ep in
                    x]
            col = "b"
        plt.figure(0)
        plt.plot(x, avgs, color=col, label=n)
        plt.title('DQN Score')
        plt.savefig(savescore)

    print(Rl.agent.Q_table)
    print('number', number)
    print('terminalNumber', terminalNumber)
    print('Goals', Rl.agent.goalsScored)
    print('Failures', Rl.agent.failures)
    # Frem til hit
