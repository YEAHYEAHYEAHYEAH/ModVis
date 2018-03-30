# -*- coding: utf-8 -*-
"""
This program is to initialise a lattice which obeys the Cahn-Hilliard equation.
"""
from __future__ import unicode_literals

#import warnings
#warnings.filterwarnings("ignore")


#       Checkpoint 3a: Cahn Hilliard
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

reload(sys)  
sys.setdefaultencoding('utf-8')



#=======================================================================
#               Simulation and Animation Section
#_______________________________________________________________________


class CahnHilliard(object):
    """
    A class used for a Cahn Hilliard Simulation model.
    """
    def __init__(self,dx,dt,a=0.1,kappa=0.1,M=0.1,size=np.array([50,50])):  # Can be instantiated with different values + defaults
        self._xDim = size[0]                                            # the system's x Dimension
        self._yDim = size[1]                                            # y Dimension
        self.lo,self.up  = self._systemRange()
        self.system   = rd.uniform(self.lo,self.up,size)                # a random array of floats between in given range (psi)
        self.muSystem = np.zeros(size, dtype=float)                     # an array of chemical potentials
        self.freeEnergy = 0
        
        self.dx,self.dt          = dx,dt
        self.a,self.kappa,self.M = a,kappa,M
        
        self.noSweeps = 0                                               
        self._totalSweeps()

    def _systemRange(self):
        while True:
            value1 = raw_input("Please enter a decimal number for the initial lower range of phi: ")
            if value1.strip("-1234567890.")=="":
                value1 = float(value1)
                if value1 <=1.0 and value1 >= -1.0:
                    print "Acceptable value.\n"
                    break
                else: print "A decimal between -1 and 1."
            else: print "A decimal between -1 and 1."
        while True:
            value2 = raw_input("Please enter a decimal number for the initial upper range of phi: ")
            if value2.strip("-1234567890.")=="":
                value2 = float(value2)
                if value2 <=1.0 and value2 >= -1.0 and value2>=value1:
                    print "Acceptable value.\n"
                    break
                else: print "A decimal between -1 and 1 and greater than lower range."
            else: print "A decimal between -1 and 1 and greater than lower range."
        return float(value1),float(value2)

    def _totalSweeps(self):
        while True:
            totSweeps = raw_input("How many sweeps would you like to be simulated? ")
            if totSweeps.strip("0123456789") == "": break
            else: print "Enter an integer number."
        self.totSweeps = int(totSweeps)

#_______________________Simulation_Methods______________________________

    def genNNList(self,x,y):
        xUp = (x+1+2*self._xDim)%self._xDim
        xDn = (x-1+2*self._xDim)%self._xDim
        yUp = (y+1+2*self._yDim)%self._yDim
        yDn = (y-1+2*self._yDim)%self._yDim                             # N,S,E,W
        NN = [[xDn,y],[xUp,y],[x,yDn],[x,yUp]]
        return NN

    def gradDotGrad(self,lattice,x,y):                                         # 1D: ∇φ(x,t) = [ φ(j+1;n) − φ(j−1;n) ] / 2δx
        if lattice == "Phi": lattice = self.muSystem
        elif lattice == "Mu": lattice = self.system
        NN = self.genNNList(x,y)
        dGrad = 0.0
        for i in xrange(2):
            i*=2
            dGrad += ((-lattice[NN[i][0]][NN[i][1]]   + \
                      lattice[NN[i+1][0]][NN[i+1][1]])/(2.0*self.dx))**2.0
        return dGrad
        
    def laplacian(self,lattice,x,y):                                       # 1D: ∇^2φ(x,t) = [ φ(j+1;n) + φ(j−1;n) − 2φ(j;n) ] /δx^2
        if lattice == "Phi": lattice = self.system
        elif lattice == "Mu": lattice = self.muSystem
        NN = self.genNNList(x,y)        
        laplLattice = -4*lattice[x][y]
        for i in NN: laplLattice += lattice[i[0]][i[1]]
        laplLattice /= (self.dx**2.0)
        return laplLattice

    def calcFreeEnergy(self):                                            # f = −a/2 φ(x,t)^2 + a/4 φ(x,t)^4 + κ/2 (∇φ(x,t))^2
        self.freeEnergy = 0
        for i in xrange(self._xDim):
            for j in xrange(self._yDim):
                self.freeEnergy += -self.a/2 * self.system[i][j]**2.0 + \
                                    self.a/4 * self.system[i][j]**4.0 + \
                                self.kappa/2 * self.gradDotGrad("Phi",i,j)
        return self.freeEnergy

    def generateMu(self):
        for i in xrange(self._xDim):                                    # μ(x,t) = −aφ(x,t)+aφ(x,t)^3−κ∇^2φ(x,t)
            for j in xrange(self._yDim):                                               
                self.muSystem[i][j]= -self.a*self.system[i][j]      +  \
                                      self.a*self.system[i][j]**3.0 -  \
                                      self.kappa*self.laplacian("Phi",i,j)
    
    def updateSystem(self):
        self.generateMu()                                               # φ(j+1;n) + φ(j−1;n) − 2φ(j;n)/δx^2 
        for i in xrange(self._xDim):
            for j in xrange(self._yDim):
                self.system[i][j] += (self.M*self.dt)*self.laplacian("Mu",i,j)

                
#_______________________Animation_Methods_______________________________

    def createCanvas(self):
        self.plotFigure, self.ax = plt.subplots()
        self.plotFigure.set_facecolor('black')
        self.ax.set_axis_bgcolor('0.05')                                # black bg without axis

    
    def drawFrame(self):                                                # method to draw a frame of animation
        self.ax.set_xlabel("Number of Sweeps: "+str(self.noSweeps) + \
                           "\nFree Energy: "+str(self.freeEnergy),color="w")
        self.ax.contourf(self.system,100,cmap=plt.cm.hot, alpha=.9, extent=(0,1,0,1))

        """
        if self.vis == 'F': self.ax.imshow(self.system,interpolation='nearest')          # draws pixels FAST
        elif self.vis == 'C': 
           # Z1 = np.add.outer(range(self._xDim), range(self._yDim)) % 2                  # chessboard
           # self.ax.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest')
           # self.ax.imshow(self.system,cmap=plt.cm.hot, alpha=.9, interpolation='sinc')  # cool looking flames (white=rec) (black=sus) (red=inf)
        """    

    def updateFrame(self,x):                                            # updates frame every xDim * yDim calculations
        updateSize = 1
        for i in xrange(updateSize):
            self.updateSystem()
        self.noSweeps += updateSize
        if self.noSweeps-self.totSweeps>0:
            exit()
        self.ax.clear()
        self.calcFreeEnergy()
        self.drawFrame()                                                # plots frame again
	
    def animateSim(self):                                               # animates dynamics
        self.createCanvas()
        ani = animation.FuncAnimation(self.plotFigure, self.updateFrame,
                                      init_func=self.drawFrame,interval=1)
        plt.show()
        
    def writeData(self):
        now = str(datetime.datetime.now())[0:-10]
        dataFile = open("CahnHilliardData/Cahn Hilliard {} Data.txt".format(now),'w')
        dataFile.write("Lattice Size: {}x{}. Initial φ Range: {} -> {}\n".format(self._xDim,self._yDim,self.lo,self.up))
        self.createCanvas()
        self.drawFrame()
        
        title = "Cahn Hilliard {} ".format(now)
        plt.savefig(title+"0 sweeps", dpi=300, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format=None,
                transparent=False, bbox_inches='tight', pad_inches=0.01,
                frameon=None)
        
        for i in xrange(self.totSweeps):
            self.updateSystem()
            self.calcFreeEnergy()
            dataFile.write("{},{}\n".format(i,self.freeEnergy))
        self.updateFrame(1)
        
        now = str(datetime.datetime.now())[0:-10]
        title = "Cahn Hilliard {} ".format(now)
        
        plt.savefig(title+"{} sweeps".format(self.totSweeps), dpi=300, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format=None,
                transparent=False, bbox_inches='tight', pad_inches=0.01,
                frameon=None)


def plotData():
    x = os.listdir(os.path.abspath("CahnHilliardData"))
    if len(x) == 0:
        print "No Data Points to compare"
        print "Please make file by using the save to file feature"
    else:
        print "Choose File"
        fileDir = os.path.abspath("CahnHilliardData")
        for i, item in enumerate(x):
            print "{} : Results on Date {}".format(i + 1, item.replace("Cahn Hilliard",""))
        while True:
            fileChose = int(raw_input("Enter number that corresponds to desired file: "))
            if fileChose > len(x):
                print "Enter a valid number inside the amount provided"
            elif fileChose < 1:
                print "Enter sensible value"
            else:
                break
        fileName = x[fileChose - 1]
        fileTxt = open(os.path.join(fileDir, fileName), 'r')
        firstLine = fileTxt.readline().replace("φ", "$\phi$")
        fileData = np.genfromtxt(os.path.join(fileDir, fileName),delimiter=',', dtype=float, skip_header=1)
        
        sweepNumber = fileData[:,0]
        freeEnergy  = fileData[:,1]

        plt.plot(sweepNumber,freeEnergy,  color="k", marker="o", linestyle=" ")
        plt.xlabel("Sweep Number")
        plt.ylabel("Free Energy")
        plt.title(u"{}".format(firstLine))
        plt.savefig(fileName[0:-4], dpi=300, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format=None,
                transparent=False, bbox_inches='tight', pad_inches=0.01,
                frameon=None)
        plt.close()

while True:
    plotOrSim = raw_input("Do you want to plot data ('p') or run a simulation ('r')? ")
    if plotOrSim.lower().strip("pr")=="": break
    else: print "Enter a 'w' or 'v'"

if plotOrSim == 'p': plotData()

elif plotOrSim == 'r':
    lattice = CahnHilliard(1,2)                               #(dx,dt)
    while True:
        animOrWrite = raw_input("Do you want to visualise ('v') or create a data file ('w')? ")
        if animOrWrite.lower().strip("wv")=="": break
        else: print "Enter a 'w' or 'v'"

    if animOrWrite.lower() == 'w': lattice.writeData()
    if animOrWrite.lower() == 'v': lattice.animateSim()
