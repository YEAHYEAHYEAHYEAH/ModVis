# -*- coding: utf-8 -*-
"""
This program is to initialise a lattice which obeys the Cahn-Hilliard equation.
"""
from __future__ import unicode_literals

import warnings
warnings.filterwarnings("ignore")


#       Checkpoint 3b: Poisson Equation
#_______________________________________________________________________

# importing modules
import sys
import os
import numpy as np
import math as m
import datetime
from numpy import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

reload(sys)  
sys.setdefaultencoding('utf-8')

def getUserInt(question):
    userInt = raw_input(question)
    if userInt.strip("0123456789") == "" and userInt!="": return int(userInt)
    else: 
        print "Enter an integer number."
        getUserInt(question)

def getUserFloat(question):
    userFloat = raw_input(question)
    if userFloat.strip("-1234567890.")=="" and userFloat!="":
        userFloat = float(userFloat)
    if value2 <=1.0 and value2 >= -1.0 and value2>=value1:
        print "Acceptable value.\n"
        break
    else: print "A decimal between -1 and 1 and greater than lower range."
    else: print "A decimal between -1 and 1 and greater than lower range."

#=======================================================================
#               Simulation and Animation Section
#_______________________________________________________________________

class Poisson(object):
    """
    A class used to simulate Poisson's Equation.
    """
    def __init__(self,dx,dt,eps=1,size=np.array([50,50,50])):  # Can be instantiated with different values + defaults
        self.size = size
        if type(size)!= np.ndarray:
            self._xD = getUserInt("System's x dimension? ")
            self._yD = getUserInt("System's y dimension? ")
            self._zD = getUserInt("System's z dimension? ")
        else:
            self._xD = size[0]                                          # the system's x Dimension
            self._yD = size[1]                                          # y Dimension
            self._zD = size[2]                                          # z Dimension

        self.elPotSys   = np.zeros((self._xD,self._yD,self._zD), dtype=float)     # initialise electric potential array as all zero
        self.chDenSys   = np.zeros((self._xD,self._yD,self._zD), dtype=float)     # initialise charge density as 1.0 at dead centre
        self.chDenSys[int(self._xD/2.0)][int(self._yD/2.0)][int(self._zD/2.0)] = -1.0
        
        self.dx,self.dt = dx,dt
        self.eps = eps
        
        self.noSweeps = 0                                               # vis='N' for none
        self.totSweeps = getUserInt("How many sweeps would you like to be simulated? ")
        self.fSlice = 0
        

#_______________________Simulation_Methods______________________________

    def genNNList(self,x,y,z):
        xUp = (x+1+2*self._xD)%self._xD
        xDn = (x-1+2*self._xD)%self._xD
        yUp = (y+1+2*self._yD)%self._yD
        yDn = (y-1+2*self._yD)%self._yD                                 # N,S,E,W
        zUp = (z+1+2*self._zD)%self._zD
        zDn = (z-1+2*self._zD)%self._zD
        NN = [[xDn,y,z],[xUp,y,z],[x,yDn,z],[x,yUp,z],[x,y,zDn],[x,y,zUp]]
        
        #xyz = [x,y,z]
        #NN2 = np.array([xyz]*6)
        #for i in xrange(3):
        #    NN2[2*i][i] = (xyz[i]-1+2*self.size[i])%self.size[i]
        #    NN2[2*i+1][i] = (xyz[i]+1+2*self.size[i])%self.size[i]
        return NN
        
    def computeAvPot(self):
        return np.sum(self.elPotSys)/(self._xD*self._yD*self._zD)
        
    def jacobiUpdate(self):
        elPotSysNew= np.zeros((self._xD,self._yD,self._zD), dtype=float)     # initialise electric potential array as all zero
        for i in xrange(1,self._xD-1):
            for j in xrange(1,self._yD-1):                              # 1 -> total-1 ,, enforces Dirichlet Boundary.
                for k in xrange(1,self._zD-1):
                    NearN = self.genNNList(i,j,k)
                    elPotSysNew[i][j][k] = self.chDenSys[i][j][k]
                    for NN in NearN:
                        elPotSysNew[i][j][k]+=self.elPotSys[NN[0]][NN[1]][NN[2]]
                    elPotSysNew[i][j][k] /= 6.0
        self.elPotSys = elPotSysNew

#_______________________Visualisation_Methods___________________________

    def createCanvas(self):
        self.plotFigure, self.ax = plt.subplots()
        self.plotFigure.set_facecolor('black')
        self.ax.set_axis_bgcolor('0.05')                                # black bg without axis

    def updateFrame(self,x):                                            # updates frame every xDim * yDim calculations
        self.ax.clear()
        
        self.drawFrame()

        self.fSlice += 1
        if self.fSlice == self._zD: plt.close('all')

    def drawFrame(self):
        self.ax.set_xlabel("Frame Slice: {}".format(self.fSlice),color="w")

        
        self.ax.contourf(self.elPotSys[self.fSlice],100,cmap=plt.cm.hot, alpha=.9, extent=(0,1,0,1),vmin=self.minP,vmax=self.maxP)
        #self.ax.imshow(self.elPotSys[self.fSlice],interpolation='nearest')             # draws pixels FAST
        
        

    def drawSlices(self):                                               # animates dynamics
        self.maxP = np.amax(self.elPotSys)
        self.minP = np.amin(self.elPotSys)
        try:
            self.fSlice = 0
            self.createCanvas()
            ani = animation.FuncAnimation(self.plotFigure, self.updateFrame,init_func=self.drawFrame,interval=1)
            plt.show()
        except:
            os.system('clear')
            print "Calculating..."
lattice = Poisson(1,1)
for i in xrange(50): 
    print i
    lattice.jacobiUpdate()

x = []
y = []
for i in xrange(lattice._zD):
    print i
    x.append(i)
    y.append(lattice.elPotSys[i][24][24])
    
plt.plot(x,y)
plt.show()

"""
for i in xrange(20):
    for i in xrange(10):
        lattice.jacobiUpdate()

"""
