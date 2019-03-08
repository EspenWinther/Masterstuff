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
import DNVGL as Rl
import numpy as np
import math
import LOS
import pickle

#
#True if need to run without starting/using the Cybersea sim
NON_CS_DEBUG = False
#True if Cybersea config should be loaded at startup or not if want to save time
LOAD_CS_CFG = False
#not in use
USER_DIR = "C:\\temp"
#ports btw python app and Cybersea sim app
WEB_SERVER_PORT_INITIAL = 8085
PYTHON_PORT_INITIAL = 25338

SCEN_PARA_FILE = "scenparafile.txt"
#run or episode counter
run_ix = 0

#makes a list of all possible combinations of the run parameter range values
def make_para_vals():
    pv = para_min[:]
    pvs = [pv[:]]
    pix = paras-1
    while pix > -1:
        pv[pix] += para_step[pix]
        if pv[pix] > para_max[pix]:
            pv[pix] = para_min[pix]
            pix -= 1
        else:
            pvs.append(pv[:])
            pix = paras-1
    
    return pvs

####
#***** EDITABLE *****
#get the next run parameter combination for the coming run
#default from ix=0 to end ("lowest" to "highest" point)
#'bwd' = from ix=end to 0
#'rand', initially randomly shuffled and then ix=0 to end.
#new ways can be implemented depending on exploration needs
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
    else:  #add new ways above !
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

    #Initialize vectors
    sims= []
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

    #Load parameters from file
    sp_reader = spreader.ScenParaReader(SCEN_PARA_FILE)
    scen_para = sp_reader.para

    #reading in the simulation init parametres
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

    #reading in the rest of the parameters
    scen_init = scen_para['scen_init']
    run_init = scen_para['run_init']
    para_name = scen_para['name']
    para_min = scen_para['min']
    para_max = scen_para['max']
    para_step = scen_para['step']

    #number of run parameters (not values)
    paras = len(para_name)

    para_vals = make_para_vals()
    #make room for the already runned runs/episode parameter values
    old_para_vals = []

    #Start up all simulators
    for sim_ix in range(NUM_SIMULATORS):
        web_port = WEB_SERVER_PORT_INITIAL + sim_ix
        python_port = PYTHON_PORT_INITIAL + sim_ix
        log("Open CS sim " + str(sim_ix) + " @ port=" + str(web_port) + " Python_port=" + str(python_port))
        sims.append(None)
        sim_initiated.append(False)
        if not NON_CS_DEBUG:
            sims[-1] = DigiTwin('Sim'+str(1+sim_ix), LOAD_CS_CFG, CS_PATH, CS_CONFIG_PATH, USER_DIR, web_port, python_port)
        sim_semaphores.append(threading.Semaphore())
        print('sim_ix')
    log("Connected to simulators and configuration loaded")

    #Initiate the scenario of all simulators (scen_init parameters)
    #if this should be done for each run/episode, place parameters in run_init

    #kommentert ut av meg

#    if len(scen_init):
#        print ('init')
#        for sim_ix in range(NUM_SIMULATORS):
#            if THREADING:
#                sim_semaphores[sim_ix].acquire()
#                log("Locking sim" + str(sim_ix+1) + "/" + str(NUM_SIMULATORS))
#
#            run_args = {'sim_init':sim_init,'run_init':scen_init}
#            run_name = 'Sim'+str(sim_ix+1)+' Run0'
#            init_runs.append(ScenarioRun(run_name, sims[sim_ix], **run_args))
#
#            if THREADING:
#                t = threading.Thread(target=init_runs[-1].run, args=[0,INIT_SIM_STEPS,SIM_STEP,sim_semaphores[sim_ix]])
#                t.daemon = True
#                t.start()
#
#            else:
#                init_runs[-1].run(0,INIT_SIM_STEPS,SIM_STEP)

    #Til hit

    #Loop through the runs
    def reset(string1, string2):
        sims[0].val(string1, string2,1.)
        sims[0].step(20)
        sims[0].val(string1, string2,0.)
        sims[0].step(50)

    def sat(value):
        num = math.ceil(value)
        return num

    sim_ix = 0
    Rl.create_states()
    
    #print(Rl.agent.Q_table)

    reset('Hull', 'StateResetOn')
    start_pos = (-1400.,1400.) #East(x) North(y)
    sims[0].val('Hull', 'PosNED[0]', 1400.) #Y north
    sims[0].val('Hull', 'PosNED[1]', -1400.) #X east
    Rl.agent.chosen_pos = (start_pos)
    reset('Hull', 'StateResetOn')

    sims[0].val('LOSGuidance', 'numWaypoints', 2.)
    sims[0].val('LOSGuidance', 'activateWaypoints', 1.)
#    sims[0].val('LOSGuidance', 'distanceNewWaypoint', 10.)

    
    running = 1
    num_episodes = 100
    
#    dirX = [20,340,160]
#    speedX = [5,5,5]
    
    # Set pid variabels 
#    LOS.pid.setKd(100.)
#    LOS.pid.setKi(0.)
#    LOS.pid.setKp(400.)
    number = 0
    Rl.agent.onland = sims[0].val('LandEstimation', 'OnLand')

    
    while running == 1:
        print('While løkka')
        if THREADING:
            sim_semaphores[sim_ix].acquire()
            log("Locking sim" + str(sim_ix+1) + "/" + str(NUM_SIMULATORS))

        for epsiodes in range(num_episodes):
            terminal = 0
            Rl.agent.terminal = 0
            Rl.agent.onland = sims[0].val('LandEstimation', 'OnLand')
            while Rl.agent.terminal !=1:
    #            Rl.update_Q(0.9,0.8)
                Rl.bestVSrandom(0.1)
    #resets if hits land, but has same position as target. this must be fixed!!!!!!!!!!!!!!!!
                print('FORLOOP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                x_pos = list(sims[0].val('Hull', 'Eta[1]'))[1]
                y_pos = list(sims[0].val('Hull', 'Eta[0]'))[0]
                Rl.agent.pos = (float(x_pos),float(y_pos)) #(list(x_pos)[1], list(y_pos)[0])
                Rl.agent.checkPos()
    #            Rl.update_Q(0.9,0.8)            
    
                reset('LOSGuidance','StateResetOn')
                sims[0].val('LOSGuidance', 'iLosWaypointsX[0]',Rl.agent.nxtpos[1])
                sims[0].val('LOSGuidance', 'iLosWaypointsY[0]',Rl.agent.nxtpos[0])
                sims[0].val('LOSGuidance', 'iLosWaypointsX[1]',Rl.agent.fakepos[1])
                sims[0].val('LOSGuidance', 'iLosWaypointsY[1]',Rl.agent.fakepos[0])
    #            sims[0].val('LOSGuidance', 'UpdateWaypoints',1)
    #            sims[0].val('LOSGuidance', 'UpdateWaypoints',0)
    #            sims[0].step(10)
    
                sims[0].val('manualControl', 'UManual',0.3)
                
                while Rl.agent.nxtpos != Rl.agent.chosen_pos and Rl.agent.terminal == 0:
                    Rl.agent.checkPos()
                    if Rl.agent.onland == 1:
                        print('LAND')
                        sims[0].val('Hull', 'PosNED[0]', 1400.) #Y north
                        sims[0].val('Hull', 'PosNED[1]', -1400.) #X east
                        Rl.agent.chosen_pos = (start_pos)
                        reset('Hull', 'StateResetOn')
                        Rl.agent.terminal = 1
                    else:
                        Rl.agent.onland = sims[0].val('LandEstimation','OnLand')
                        print('scenmanloop reward',Rl.agent.reward)
                        print(Rl.agent.onland, sims[0].val('LandEstimation', 'OnLand'))
                            
                        x_pos = list(sims[0].val('Hull', 'Eta[1]'))[1]
                        y_pos = list(sims[0].val('Hull', 'Eta[0]'))[0]
                        Rl.agent.pos = (x_pos, y_pos)
    #                    sims[0].val('manualControl', 'PsiManual',heading_d)
                        sims[0].step(10)
                        Rl.agent.checkPos()
                        sims[0].val('LOSGuidance', 'UpdateWaypoints',1)
                        sims[0].step(10)
                        sims[0].val('LOSGuidance', 'UpdateWaypoints',0)
                        sims[0].step(10)
                        
                
                Rl.update_Q(0.9,0.8)
                number = number+1
                print('number',number)
                reset('LOSGuidance','StateResetOn')
                print(Rl.agent.Q_table)
    
            running = 0
        
        pickle.dump(Rl.agent.Q_table, open('progress.p', 'wb')) #dumping the Qtable variable
            # Frem til hit
        #pickleing
#        pickle.dump(Qtable, open('progress.p', 'wb')) #dumping the Qtable variable
#       Qtable = pickle.load(open('progress.p', 'rb') #loading the Qtable variable

# Fra her

        #make args for the simulation
#        run_args = {'sim_init':sim_init,'run_init':run_init}
#
#        para_val = get_next_para_val()
#        if para_val == None:  #No more or failed to get a new
#            break  #end simulation
#            
#        #special for scenario where run_para name has an 'X' at the end ex 'dirX'
#        #X is increased for each run looping through active target ships
#        for pix in range(paras):
#            if 'X' in para_name[pix]:
#                pn = para_name[pix]
#                
#                #CS has only X target ships available, so if the X+1 ship is needed, 
#                #reuse the 0th, and so on
#                tsix = run_ix+1
#                while tsix > MAXACTIVETARGETSHIPS:
#                    tsix -= MAXACTIVETARGETSHIPS
#                 
#                pn = pn.replace("X",str(tsix))
#                run_args[pn] = para_val[pix]
#            else:
#                run_args[para_name[pix]] = para_val[pix]
#        
#        #name the run something
#        run_name = 'Sim'+str(sim_ix+1)+' Run'+str(run_ix+1) + ' ' + str(para_val)
#        #create a new run object and add to runs list
#        runs.append(ScenarioRun(run_name, sims[sim_ix], **run_args))

# Til her: kan fjernes, brukes ikke

#        if THREADING:
##TODO remove the CONSTANTS, they are in para_args
#            t = threading.Thread(target=runs[-1].run, args=[0,SIM_STEPS,SIM_STEP,sim_semaphores[sim_ix]])
#            t.daemon = True
#            t.start()
#
#        else:
##TODO remove the CONSTANTS, they are in para_args
#            runs[-1].run(0,SIM_STEPS,SIM_STEP)
#
#        #distribute to each simulator instance
#        if NUM_SIMULATORS > 1:
#            sim_ix += 1
#            if sim_ix >= NUM_SIMULATORS:
#                sim_ix = 0
#        #count runs/episodes
#        run_ix += 1

#  