#       Checkpoint 2b: SIRS Model
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


### 0 = susceptible
### 1 = infected
### 2 = recovered

#=======================================================================
#               Simulation and Animation Section
#_______________________________________________________________________


class SIRSModel(object):
    """
    A super class used for different representations of the IsingModel.
    """
    def __init__(self,uniqID,probs,size = np.array([100,100]),vis="C",immunityFrac=0.0):# Can be instantiated with different values + defaults
        self._xDim = size[0]                                            # the systems x Dimension
        self._yDim = size[1]                                            # y Dimension
        self._system   = rd.random_integers(0,2,size)                   # a random array of 0s, 1s, and 2s of specified size
        self._uniqueId = uniqID                                         # an object identifier
        self.vis = vis                                                  # chooses between Fast 'F' or cool 'C' visualisation
        self.noSweeps = 0                                               # vis='N' for none, only records total infected.
        self.computeLife()
        self.probSI = probs[0]
        self.probIR = probs[1]
        self.probRS = probs[2]
        if immunityFrac!=0.0: self.makeSomeImmune(immunityFrac)         # makes a certain fraction immune
    
    def makeSomeImmune(self,frac):
        """
        A function to make a certain fraction of the lattice sites
        'immune' to the spread of the infection.
        :frac: a float between 0.0 and 1.0.
        """
        noImmune = int(frac*self._xDim*self._yDim)                      # integer number of immune sites.
        for i in xrange(noImmune):
            rCo = self.getRndCoords()
            condition = False
            while self.system[rCo[0]][rCo[1]] == 3:                     # to account for the same random coord being generated
                rCo = self.getRndCoords()                               # which leads to less than immune frac being immune
            self.system[rCo[0]][rCo[1]] = 3


#_______________________Object_Oriented_________________________________

    @property                                                           # getter + setter methods for system
    def system(self):        return self._system
    @system.setter
    def system(self,newSys): self._system = newSys

    def __str__(self):    return str(self._system)                      # prints the system as array of 0s, 1s, 2s.

        
#_______________________Simulation_Methods______________________________
      
    def computeLife(self):                                              # method to compute the total number of S,I & R sites.
        self.noRec = 0
        self.noInf = 0
        self.noSus = 0
        for i in xrange(self._xDim):                                    
            for j in xrange(self._yDim):
                val = self.system[i][j]
                if val ==1:  self.noInf +=1
                if self.vis!='N':                                       # for data measurements where only number infected is recorded.
                    if val ==2:  self.noRec +=1                         # if wanting to animate and display counters
                    if val ==0:  self.noSus +=1



    def getRndCoords(self):                                             # method to select a random position in the array
        xRdm = rd.randint(0,self._xDim)
        yRdm = rd.randint(0,self._yDim)
        coords = np.array([xRdm,yRdm])
        return xRdm,yRdm
    
    def rndProb(self): return rd.random()  

    def dynamics(self):                                                 # 0 = susceptible # 1 = infected # 2 = recovered
        sys = self.system
        for _x in xrange(self._xDim):                                   # live cells live with 2 or 3 NN
            for _y in xrange(self._yDim):                               # dead cells become alive with 3 NN
                i,j = self.getRndCoords()
                if sys[i][j] == 0:                                      
                    if self.atLeastOneInf(i,j):                         # if there is at least one infected & greater than random prob, 
                        if self.probSI > self.rndProb():                # susceptible goes to infected
                            sys[i][j] = 1
                            break
                elif sys[i][j] == 1:                                    # infected goes to recovered if greater than random prob
                    if self.probIR > self.rndProb():
                        sys[i][j] = 2
                        break
                elif sys[i][j] == 2:
                    if self.probRS > self.rndProb():                    # recovered goes susceptible if greater than random prob
                        sys[i][j] = 0
                        break
        
    def atLeastOneInf(self,x,y):                                        # method to see if there is at least one infected site as NN. BOOLEAN
        sys = self.system
        xUp = x + 1
        yUp = y + 1                                                     # N,S,E,W
        xDn = x - 1
        yDn = y - 1
        if xUp == self._xDim: xUp=0
        if yUp == self._yDim: yUp=0
        if xDn == -1: xDn = self._xDim-1                                # periodic boundary conditions
        if yDn == -1: yDn = self._yDim-1
        NNList = [[xDn,y],[xUp,y],[x,yDn],[x,yUp]]
        for co in NNList:
            if sys[co[0]][co[1]] == 1: return True                      # returns true if there is one infected
        
        return False                                                    # else false


#_______________________Animation_Methods_______________________________

    def createCanvas(self):
        self.plotFigure, self.ax = plt.subplots()
        self.plotFigure.suptitle("Ising Model Dynamics")                # creating axis and plots for plotting
        self.plotFigure.set_facecolor('black')
        self.ax.set_axis_bgcolor('0.05')                                # black bg without axis

    
    def drawFrame(self):                                                # method to draw a frame of animation    
        self.ax.set_xlabel('Total Recovered (W): '+str(self.noRec)+
                           '\nTotal Infected (R): '+str(self.noInf)+
                           '\nTotal Susceptible (B): '+str(self.noSus)+
                           '\nTotal number of sweeps: '+str(self.noSweeps),color="w")    # draws all the labels

        if self.vis == 'F': self.ax.imshow(self.system,interpolation='nearest')          # draws pixels FAST
        elif self.vis == 'C': 
            Z1 = np.add.outer(range(self._xDim), range(self._yDim)) % 2                  # chessboard
            self.ax.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest')
            self.ax.imshow(self.system,cmap=plt.cm.hot, alpha=.9, interpolation='sinc')  # cool looking flames (white=rec) (black=sus) (red=inf)
            

    def updateFrame(self,x):                                            # updates frame every xDim * yDim calculations
        self.dynamics()
        self.computeLife()
        self.noSweeps += 1
        self.ax.clear()
        self.drawFrame()                                                # plots frame again
	
    def animateSim(self):                                               # animates dynamics
        self.createCanvas()
        ani = animation.FuncAnimation(self.plotFigure, self.updateFrame,
                                      init_func=self.drawFrame,interval=1)
        plt.show()
        


#=======================================================================
#                  Program Modes Section
#_______________________________________________________________________



## probSI = p1, probIR = p2, probRS = p3   ... SIRSModel(0, [p1,p2,p3])


#=======================================================================
#..................Contour Plot, fixed p2=0.5...........................

def contourData():
    """
    Mode which writes data files containing p1,p3 on top line and the
    numbers of total infected below. p increments of 0.1 from 0 -> 1.0.
    """
    dataFile = open("ContourPlotData.txt",'w')
    dataFile.write("Contour Plot Data\n\nS->I Probability,R->S Probability\nTotal Infected\n\n")
    for i in xrange(11):

        prbSI = i*0.1
        for j in xrange(11):
            print "p1: "+str(float(i/10.))+"   p3: "+str(float(j/10.0))
            totalInf = []
            prbRS = j*0.1
            dataFile.write("{},{}\n".format(prbSI,prbRS))
            game = SIRSModel(0, [prbSI,0.5,prbRS], vis='N')
            for _equib in xrange(400):                                  # lets the simulation come to equilibrium
                game.dynamics()
            for _data in xrange(1000):                                  # calculates and records data
                game.dynamics()
                game.computeLife()
                dataFile.write("{},".format(game.noInf))
            dataFile.write("\n")
    dataFile.close()
    

def plotContourData(variance=False):
    fileStream = open("ContourPlotData.txt", "r")
    lnNo = 0
    contourArray = np.zeros((11,11))
    for _ in xrange(5):
        next(fileStream)
    ij = []
    for line in fileStream:
        lnNo+=1
        if ((lnNo+1)%2==0):
            ij = [int(float(s)*10) for s in line.strip("\n").split(",")]
        if (lnNo%2==0):
            vals = [float(s) for s in line.strip(" \n")[0:-1].split(",")]
            
            if variance==True:
                contourArray[ij[1]][ij[0]] = np.var(np.array(vals)/2500.0)
            
            else: 
                contourArray[ij[1]][ij[0]] = np.mean(np.array(vals)/2500.0)
    fileStream.close()
    
    Z1 = np.add.outer(range(10), range(10)) % 2  # chessboard
    plt.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest', extent=(0,1,0,1))
    plt.contourf(contourArray,100,cmap=plt.cm.hot, alpha=.9, extent=(0,1,0,1))
    plt.colorbar()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.ylabel("Probability of Rec->Sus (p3)")
    plt.xlabel("Probability of Sus->Inf (p1)")
    plt.title("Average Fraction Infected (Inf to Rec p2=0.5)")
    if variance==True: plt.title("Variance of Fraction Infected (Inf to Rec p2=0.5)")
    plt.show()
        

#=======================================================================
#..................Minimum Immunity Fraction............................

def immunityData():
    dataFile = open("immunityData.txt",'w')
    dataFile.write("Fraction Immune\nTotal Infected\n\n")
    for i in xrange(51):
        fracImmune = i*0.02
        print "The program is {}% done".format((i/51.0)*100)
        dataFile.write(str(fracImmune)+"\n")
        game = SIRSModel(0, [0.5,0.5,0.5],immunityFrac=fracImmune)
        
        died=False
        
        for _ in xrange(500):
            game.dynamics()
            game.computeLife()
            if game.noInf == 0:
                died = True
                dataFile.write(str(game.noInf)+",")
                break
        if died ==False:
            for _ in xrange(5000):
                game.dynamics()
                game.computeLife()
                dataFile.write(str(game.noInf)+",")
                if game.noInf==0: break
        dataFile.write("\n")
    dataFile.close()


def plotImmunityData():
    fileStream = open("immunityDataGood.txt", "r")
    lnNo = 0
    fracs = []
    avInf = []
    avInfErr = []

    for _ in xrange(3):
        next(fileStream)
    for line in fileStream:
        lnNo+=1
        if ((lnNo+1)%2==0):
            immuneFrac = line.strip("\n")
            descriptor = immuneFrac[0:4]
            fracs.append(float(immuneFrac))
        if (lnNo%2==0):
            vals = [float(s)/2500.0 for s in line.strip(" \n")[0:-1].split(",")]
            
            if 0 in vals: 
                avInf.append(0)
                avInfErr.append(0)
            else:
                avInf.append(np.mean(vals))
                avInfErr.append(np.std(vals)/(len(vals)**0.5))
    fileStream.close()
    
    plt.errorbar(fracs,avInf,yerr=avInfErr,color="k",linestyle=" ",marker=".")
    plt.xlabel("Immunity Fraction")
    plt.ylabel("Average Fraction Infected")
    plt.title("Minimum immunity fraction to prevent spread: p1=p2=p3=0.5")
    plt.show()


#=======================================================================
#..................Main Simulation Mode.................................

"""
Lots of functions to enable the program to run in interactive mode, with
appropriate sanitisation of inputs and looping when wrong inputs given.
Self explanatory as to what they do.
"""

def giveMeAValue(bitOfString):
    """
    Method which gets a value for a probability and sanitises it.
    """
    value = raw_input("Please give a value for the {} probability: ".format(bitOfString))
    if value.strip("1234567890.")=="":
        value = float(value)
        if value <=1.0 and value >= 0.0:
            print "Acceptable value.\n"
            return value

        else: giveMeAValue(bitOfString)
    else: giveMeAValue(bitOfString)


def mainQuestions():
    """
    Mode which asks the user for values of the different probabilities
    and then animates the simulation.
    """
    print ("\nWelcome to a SIRS model simulation.\n\n")
    probSI = giveMeAValue("susceptible to infected")
    probIR = giveMeAValue("infected to recovered")
    probRS = giveMeAValue("recovered to suseptible")
    print "Here is the simulation."
    game = SIRSModel(0, [probSI,probIR,probRS],immunityFrac=0.0,vis='C')
    game.animateSim()
    game.plotResults()



#=======================================================================
#..................Main Simulation Mode.................................

#mainQuestions()



#=======================================================================
#..................Data Generation......................................

#immunityData()
#contourData()



#=======================================================================
#..................Data Plotting........................................

#plotContourData()
#plotContourData(variance=True)
#plotImmunityData()
