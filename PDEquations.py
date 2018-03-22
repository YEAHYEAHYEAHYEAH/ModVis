"""
This program is to initialise a lattice which obeys the Cahn-Hilliard
equation.
"""


#       Checkpoint 3a: PDEquations
#_______________________________________________________________________

# importing modules
import sys
import numpy as np
import math as m
from numpy import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.ticker as ticker


#=======================================================================
#               Simulation and Animation Section
#_______________________________________________________________________


class CahnHilliard(object):
    """
    A super class used for different representations of the IsingModel.
    """
    def __init__(self,dx,dt,a,kappa,size=np.array([100,100]),M=0.1,vis=True):  # Can be instantiated with different values + defaults
        self._xDim = size[0]                                            # the system's x Dimension
        self._yDim = size[1]                                            # y Dimension
        self.system   = rd.uniform(0,1,size)*2 -1                       # a random array of floats between -1, and 1.
        self.muSystem = np.zeros(size, dtype=float)                     # an array of chemical potentials
        
        self.dx,self.dt          = dx,dt
        self.a,self.kappa,self.M = a,kappa,M
        
        self.vis = vis                                                  # chooses between animation or a fixed number of runs
        self.noSweeps = 0                                               # vis='N' for none
        self._totalSweeps()
        
    def _totalSweeps(self):
        if self.vis == False:
            while True:
                totSweeps = raw_input("How many sweeps (int) would you like to be simulated? ")
                if totSweeps.strip("0123456789") == "": break
                else: print "Enter an integer number."
            self.totSweeps = totSweeps
            

    def gradPhi(x,y):                                                   # 1D: ∇^2φ(x,t) = φ(j+1;n) + φ(j−1;n) − 2φ(j;n)/δx^2
        sys = self.system
        xUp = x + 1
        yUp = y + 1                                                     # N,S,E,W
        xDn = x - 1
        yDn = y - 1
        if xUp == self._xDim: xUp=0
        if yUp == self._yDim: yUp=0
        if xDn == -1: xDn = self._xDim-1                                # periodic boundary conditions
        if yDn == -1: yDn = self._yDim-1
        NN = [[xDn,y],[xUp,y],[x,yDn],[x,yUp]]
        gradPhi = -4*sys[x][y]
        for i in NN:
            gradPhi += sys[i]
        gradPhi /= self.dx**2.0
        return gradPhi

    def generateMu(self):
        for i in xrange(self._xDim):                                    # μ(x,t) = −aφ(x,t)+aφ(x,t)^3−κ∇^2φ(x,t)
            for j in xrange(self._yDim):                                # φ(j+1;n) + φ(j−1;n) − 2φ(j;n)/δx^2
                self.muSystem[i][j] = -self.a*self.system[i][j] + \
                                       self.a*self.system[i][j]**3.0 - \
                                       self.kappa*self.gradPhi()
    
    def updateSystem(self):
        
lattice = CahnHilliard(1,1,1,1,vis=False)
