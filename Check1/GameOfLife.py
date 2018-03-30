#       Checkpoint 2a: Game of Life Model
#_______________________________________________________________________

# importing modules
import sys
import numpy as np
import math as m
from numpy import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import scipy.odr as odr
from scipy import misc


#=======================================================================
#               Simulation and Animation Section
#_______________________________________________________________________


class GameOfLife(object):
    """
    A super class used for different representations of the IsingModel.
    """
    def __init__(self,uniqID,mode,size = np.array([50,50]),vis="F"):# Can be instantiated with different values + defaults
        self._xDim = size[0]                                            # the systems x Dimension
        self._yDim = size[1]                                            # y Dimension
        self._generateSystem(mode,size)                                 # a random array of 1s and -1s of specified size
        self._uniqueId = uniqID                                         # an object identifier
        self.vis = vis
        self.mode = mode
        self.noSweeps = 0
        self.computeLife()

    
    def _generateSystem(self,mode,size):
        
        if mode=="R": self._system   = rd.random_integers(1,2,size)*2 -3              # a random array of 1s and -1s of specified size

        elif mode=="P":                                                 # system which is loaded from a 50x50 pixel picture (drawn in paint)
            im = misc.imread("pics/InitialConditionPulsar2.pgm")
            im = np.array(im)
            self._system = np.zeros(size)
            for i in xrange(self._xDim):
                for j in xrange(self._yDim):
                    if im[i][j] ==255: self._system[i][j] =-1
                    if im[i][j] !=255: self._system[i][j] = 1

        elif mode=="O":                                                 # oscillating system
            interval = 5
            self._system= np.ones(size)-2
            for j in xrange(1,int(size[0]/6)):
                for i in xrange(3):
                    for k in xrange(10):
                        self._system[j*6][i+interval*k] = 1
    
        elif mode =="G":                                                # glider system
            self._system = np.ones(size)-2
            glider = [[1, 1, 1],[1, -1, -1],[-1, 1, -1]]
            self._system =  self.smasher(glider)
            self.dataFile = open("GliderDataNew.txt",'w')
            self.dataFile.write("Glider Movement Data\nFrame,Centre of Mass Position\n\n")
 
    def smasher(self,shape):
        X = np.zeros((self._xDim, self._yDim))-1
        X[:len(shape), :len(shape[0])] = shape
        return X
        
    def centreOfMass(self):                                             # finds the centre of mass of glider
        runningTot = 0.0
        noPts = 0.0
        for i in xrange(self._xDim):
            for j in xrange(self._yDim):
                if self.system[i][j]==1:
                    noPts+=1.
                    runningTot+= (i**2.0 + j**2.0)**.5
        return runningTot/noPts
    
#_______________________Simulation_Methods______________________________
      
    def computeLife(self):                                              # method to compute the total number of ups and downs
        if self.vis=='N': return np.sum(self.system)                    # if wanting to run quickly (for measurementsn not vis)
        self.noAlive = 0
        self.noDead = 0
        for i in xrange(self._xDim):                                    # if wanting to animate and display counters
            for j in xrange(self._yDim):
                val = self.system[i][j]
                if val ==1: self.noAlive +=1
                if val ==-1: self.noDead +=1
        return self.noAlive - self.noDead

    @property                                                           # getter + setter methods for system
    def system(self):        return self._system
    @system.setter
    def system(self,newSys): self._system = newSys

    def __str__(self):    return str(self._system)                      # prints the system as array of 1s and -1s

    def sysElmnt(self,coord):                                           # method to call a value from a coordinate
        return self.system[coord[0]][coord[1]]

    def getRndCoords(self):                                             # method to select a random position in the array
        xRdm = rd.randint(0,self._xDim)
        yRdm = rd.randint(0,self._yDim)
        coords = np.array([xRdm,yRdm])
        return coords

    def dynamics(self):
        newArray = np.zeros((self._xDim,self._yDim),dtype=int)
        sys = self.system
        for i in xrange(self._xDim):                                    # live cells live with 2 or 3 NN
            for j in xrange(self._yDim):                                # dead cells become alive with 3 NN
                NearNTot = self.sumOfNN(i,j)
                if sys[i][j] == 1:                                      # if alive
                    if NearNTot==-4 or NearNTot ==-2:                   # 2 or 3 live NN
                        newArray[i][j] = 1                              # stays alive
                    else: newArray[i][j] = -1                           # if not then dies
                else:                                                   
                    if NearNTot == -2:                                  # if dead, w/ 3 live neighbours
                        newArray[i][j] = 1                              # comes alive 
                    else: newArray[i][j] = -1                           # stays dead
        self.system = newArray
        
    def sumOfNN(self,x,y):
        sys = self.system
        total = -sys[x][y]
        for i in xrange(3):
            for j in xrange(3):
                co1 = x-1+i
                co2 = y-1+j
                if co1 < 0: co1 = self._xDim-1                          # periodic boundaries
                elif co1 > self._xDim-1: co1 = 0
                if co2 < 0: co2 = self._yDim-1
                elif co2 > self._yDim-1: co2 = 0
                
                total += sys[co1][co2]                                  # -8=0Alive, -6=1Alive ... 8=8Alive 
        return total


#_______________________Animation_Methods_______________________________

    def createCanvas(self):
        self.plotFigure, self.ax = plt.subplots()
        self.plotFigure.suptitle("Ising Model Dynamics")                # creating axis and plots for plotting
        self.plotFigure.set_facecolor('black')  
        self.ax.set_axis_bgcolor('0.05')                                # black bg without axis
        #self.ax.set_ylim(0,self._yDim+1)
        #self.ax.set_xlim(0,self._xDim+1)
    
    def drawFrame(self):                                                # method to draw a frame of animation    
        self.ax.set_xlabel('Total Alive: '+str(self.noAlive)+
                           ' Total Dead: '+str(self.noDead)+
                           '\nTotal number of sweeps: '+str(self.noSweeps),color="w")   # draws all the labels

        if self.vis == 'F': self.ax.imshow(self.system,interpolation='nearest')        # draws pixels FAST
        elif self.vis == 'C': 
            
            Z1 = np.add.outer(range(self._xDim), range(self._yDim)) % 2  # chessboard
            self.ax.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest')
            self.ax.imshow(self.system,cmap=plt.cm.bone, alpha=.8, interpolation='sinc')

    def updateFrame(self,x):                                            # updates frame every xDim * yDim calculations
        self.dynamics()
        self.computeLife()
        self.noSweeps += 1
        self.ax.clear()
        self.drawFrame()                                                # plots frame again
        if self.mode == "G":
            self.dataFile.write("{},{}\n".format(self.noSweeps,self.centreOfMass()))
            if self.noSweeps == 3000:
                self.dataFile.close()
                exit()
	
    def animateSim(self):                                               # animates dynamics
        self.createCanvas()
        ani = animation.FuncAnimation(self.plotFigure, self.updateFrame,
                                      init_func=self.drawFrame,interval=1)

        plt.show()
        
    


#=======================================================================
#                  Program Modes Section
#_______________________________________________________________________

    
def fixBC(data):
    newData = np.zeros(len(data))
    fixSize = 70.71
    restTime = 0
    for i in xrange(len(data)-1):
        restTime += 1
        if data[i+1]>data[i] and restTime > 10:
            restTime = 0
            fixSize  += 70.71
        newData[i] = data[i]-fixSize
    return newData
    
def str8Line(A,x): return A[0]*x + A[1]    

def fitLine(data):
    start = 0
    end   = 0
    rest  = 0
    listOfSpeeds = []
    errors = []
    frameNo      = []
    metEnd = False
    for i in xrange(len(data)-1):
        rest+=1
        frameNo.append(i)
        if data[i+1]<data[i] and metEnd == False: 
            start = i+9
            metEnd = True
        if data[i+1]>data[i] and rest >10 :
            metEnd = False
            rest=0
            end = i
            x = frameNo[start:end]
            y = data[start:end]
            myodr = odr.ODR(odr.Data(frameNo[start:end],data[start:end]), odr.Model(str8Line), beta0=[-0.5, 70])
            result = myodr.run()
            fitLine = result.beta[0]*frameNo[start:end] + result.beta[1]
            listOfSpeeds.append(result.beta[0])
            errors.append(result.sd_beta[0])
    return np.mean(listOfSpeeds),np.mean(errors)
    
def doMeasure():
    dataFile = open("GliderData.txt",'r')
    plotArray = np.genfromtxt(dataFile,delimiter=',',skip_header=3)        
    frameNo = plotArray[:,0]
    comPos = plotArray[:,1]
    avSpeed = fitLine(comPos)
    plt.text(0,68,"Average speed: {0:.5f}".format(avSpeed[0]) + "({0:.0f}) pos/fr".format(avSpeed[1]*100000.0))
    plt.plot(frameNo,comPos,  color="k", marker=".", linestyle=" ")
    plt.xlabel("Frame Number")
    plt.ylabel("Centre of Mass Position")
    plt.title("Glider Position")
    plt.savefig("Fig", dpi=300, facecolor='w', edgecolor='w',
            orientation='landscape', papertype=None, format=None,
            transparent=False, bbox_inches='tight', pad_inches=0.01,
            frameon=None)
    plt.show()
    plt.close()

def main(mode):
    game = GameOfLife(0,mode,vis='C')
    game.animateSim()


def mainQuestions():
    hello = raw_input("\nPlease indicate whether you wish to play a random ('R'), oscillating ('O'), gliding ('G'), or preset ('P') Game Of Life?\n\n")
    if   (hello=="R"): main(hello)
    elif (hello=="O"): main(hello)
    elif (hello=="G"): main(hello)
    elif (hello=="P"): main(hello)
    else: mainQuestions()

mainQuestions()
#doMeasure()
