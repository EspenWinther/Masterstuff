# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:19:45 2019

@author: Espen Eilertsen
"""

import math
import DNVGL
import numpy as np
import digitwin
import math


class LOS:
    def __init__(self):
        #Position
        self.x = 0.1
        self.y = 0.1
        
        #Previous waypoint
        self.x_k = 0
        self.y_k = 0 
        
        #Next waypoint
        self.x_kp1 = 0.1
        self.y_kp1 = 0.1
        
        #LOS target
        self.x_los = 0
        self.y_los = 0
        
        self.R = 10000
    
    def getValues(self,x,xk,xkp1,y,yk,ykp1):
                #Position
        self.x = x
        self.y = y
        
        #Previous waypoint
        self.x_k = xk
        self.y_k = yk
        
        #Next waypoint
        self.x_kp1 = xkp1
        self.y_kp1 = ykp1
        
    def LOSG(self,x1,x2,x3,y1,y2,y3): #current values = x,xk,xkp1,y,yk,ykp1
        self.getValues(x1,x2,x3,y1,y2,y3)
    
        self.y_delta = self.y_kp1 - self.y_k
        self.x_delta = self.x_kp1 - self.x_k
        
        if self.x_delta != 0:
            self.d = self.y_delta/self.x_delta
            self.g = self.y_k -self.d*self.x_k
            
            self.b = 2*(self.d*self.g-self.d*self.y-self.x)
            self.a = 1+self.d**2
            self.c = self.x**2+self.y**2+self.g**2-2*self.g*self.y-self.R**2
            
            self.y_los = self.d*(self.x_los-self.x_k)+self.y_k
            
            if self.x_delta > 0:
                self.x_los = (-self.b + np.sqrt(self.b**2 -4*self.a*self.c))/(2*self.a)
            elif self.x_delta < 0:
                self.x_los = (-self.b - np.sqrt(self.b**2 -4*self.a*self.c))/(2*self.a)
                
            
        elif self.x_delta == 0:
            self.x_los = self.x_k
            if self.y_delta > 0:
                self.y_los = self.y+np.sqrt(self.R**2-(self.x_los-self.x)**2)
            elif self.y_delta < 0:
                self.y_los = self.y-np.sqrt(self.R**2-(self.x_los-self.x)**2)
            else:
                self.y_los = self.y_k

            
        
        self.heading_d = np.arctan2(self.y_los-self.y, self.x_los-self.x)
        
        return self.heading_d


class PID:
    """
    Discrete PID control
    """

    def __init__(self, P=0.0, I=0.0, D=0.0, Derivator=0, Integrator=0, Integrator_max=math.pi, Integrator_min=-math.pi):

        self.Kp=P
        self.Ki=I
        self.Kd=D
        self.Derivator=Derivator
        self.Integrator=Integrator
        self.Integrator_max=Integrator_max
        self.Integrator_min=Integrator_min

        self.set_point=0.0
        self.error=0.0

    def update(self,current_value):
        """
        Calculate PID output value for given reference input and feedback
        """
        print('error',self.error)
        self.error = self.set_point - current_value #Traded places 

        self.P_value = self.Kp * self.error
        self.D_value = self.Kd * ( self.error - self.Derivator)
        self.Derivator = self.error

        self.Integrator = self.Integrator + self.error

        if self.Integrator > self.Integrator_max:
            self.Integrator = self.Integrator_max
        elif self.Integrator < self.Integrator_min:
            self.Integrator = self.Integrator_min

        self.I_value = self.Integrator * self.Ki

        PID = self.P_value + self.I_value + self.D_value

        return PID

    def setPoint(self,set_point):
        """
        Initilize the setpoint of PID
        """
        self.set_point = set_point
        self.Integrator=0
        self.Derivator=0

    def setIntegrator(self, Integrator):
        self.Integrator = Integrator

    def setDerivator(self, Derivator):
        self.Derivator = Derivator

    def setKp(self,P):
        self.Kp=P

    def setKi(self,I):
        self.Ki=I

    def setKd(self,D):
        self.Kd=D

    def getPoint(self):
        return self.set_point

    def getError(self):
        return self.error

    def getIntegrator(self):
        return self.Integrator

    def getDerivator(self):
        return self.Derivator
    

    
pid = PID()
pid_pos = PID()
pid_pos1 = PID()
los = LOS()