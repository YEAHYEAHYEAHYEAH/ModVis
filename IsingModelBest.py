#       Checkpoint 1: Monte Carlo & Ising Model
#_______________________________________________________________________

# importing modules
import sys
import numpy as np
import math as m
from numpy import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors

# global arrow colour dictionary, hex code
arrCol = dict([(-1, '#ff6600'), (1, '0.4')])



class IsingModel(object):
    """
    A super class used for different representations of the IsingModel.
    """
    def __init__(self,uniqID,size = np.array([50,50]),tmp =2.3,vis='N'):# Can be instantiated with different values + defaults
        self._system   = rd.random_integers(1,2,size)*2 -3              # a random array of 1s and -1s of specified size
        self._uniqueId = uniqID                                         # an object identifier
        self._xDim = size[0]                                            # the systems x Dimension
        self._yDim = size[1]                                            # y Dimension
        self._J  = 1
        self._kB = 1                                                    # time units -- these vals = 1
        self._T  = tmp                                                  # sets temp from command line arg
        
        self.totalSwitches = 0                                          # some counters to display extra info when plotting
        self.noSweeps = 0
        self.computeUpDn()                                              # compute and initialise the magnetisation and
        self.computeTotalE()                                            # and total energy values of WHOLE system
                
        self.randProb = rd.random()                                     # a random probability between 0 and 1 to decide whether to accept flips
        self.vis = vis
        self.setType()

    def setType(self): pass
        
    def computeUpDn(self):
        self.noUp = 0
        self.noDn = 0
        for i in xrange(self._xDim):
            for j in xrange(self._yDim):
                val = self.system[i][j]
                if val ==1: self.noUp +=1
                if val ==-1: self.noDn +=1
        return self.noUp - self.noDn

    @property                                                           # getter + setter methods for system and temperature
    def system(self):
		return self._system
    @system.setter
    def system(self,newSys):
        self._system = newSys
		    
    @property
    def T(self):
        return self._T
    @T.setter
    def T(self, T):
        self._T = T

		
    def __str__(self):                                                  # prints the system as array of 1s and -1s
		return str(self._system)

		
    def sysElmnt(self,coord):                                           # method to call a value from a coordinate
        return self.system[coord[0]][coord[1]]


    def getRndCoords(self):                                             # method to select a random position in the array
        xRdm = rd.randint(0,self._xDim)
        yRdm = rd.randint(0,self._yDim)
        coords = np.array([xRdm,yRdm])
        return coords


    def computeTotalE(self):                                            # computes the total E of the system
        sys = self.system
        totalE = 0
        for i in xrange(self._xDim):
            for j in xrange(self._yDim):
                iup = i+1                                               # periodically cycles through every value
                if i == self._xDim-1: iup=0                             # avoids double counting
                jup = j+1
                if j == self._yDim-1: jup=0
                totalE += ( -self._J * sys[i][j] * 
                          (sys[iup][j]+sys[i][jup]) )                   # adds two pairwise intercations to list at once
        self.totalE = totalE
        return totalE


    def switch(self,rdCo1):                                             # method which inverts an element of the system (1 --> -1 etc)
        if self.vis!='N':
            val = self.system[rdCo1[0]][rdCo1[1]]
            if val == -1:
                self.noUp +=1
                self.noDn +=-1
            if val == 1:
                self.noUp+=-1
                self.noDn+=1
            self.totalSwitches +=1
        self.system[rdCo1[0]][rdCo1[1]] = -self.system[rdCo1[0]][rdCo1[1]]

    
    def computeDltaE(self,coOrd):                                       # finds the energy difference if one element is switched
        sys = self.system
        xCo = coOrd[0]                                                  #  (N S E W) like compass
        yCo = coOrd[1]                                                  # only considers nearest neighbours above below, left right
        xUp = xCo + 1
        yUp = yCo + 1
        xDn = xCo - 1
        yDn = yCo - 1
        if xUp == self._xDim: xUp=0
        if yUp == self._yDim: yUp=0
        if xDn == -1: xDn = self._xDim-1                                # periodic boundary conditions
        if yDn == -1: yDn = self._yDim-1
        localDeltaE  = 2*(self._J * sys[xCo][yCo] *                     # DelE = 2 J * S_i + (S_before + S_after + S_above + S_below)
                       (sys[xUp][yCo]+
                        sys[xDn][yCo]+
                        sys[xCo][yUp]+
                        sys[xCo][yDn]))
        return localDeltaE

    def dynamics(self):  pass                                           # method to be filled by child classes
        

    def calcHeatCap(self):
        sweepSize = self._xDim*self._yDim                               # define how big a sweep is
        heatCapFile = open("HeatCapacity{}.txt".format(self.dynamicsType),'w')
        heatCapFile.write("Heat Capacity Measurements \n \n\
The size of the system:                     {}\n\
The number of sweeps to equilibriate:       300\n\
The number of measurements per temperature: 1000\n\n\
================================================\n\
Temperature \nTotal Energy Values\n\
================================================\n\n".format(sweepSize))

        writeEvery = 10*sweepSize
        
        for i in range(10,30):                                          # loop through temps from 0.1 -> 3.0
            self.T=(i*0.1)
            heatCapFile.write(str(self.T)+"\n")                         # writes T on top line
            
            for j in range(500*sweepSize):                              # equilibrate the system
                self.dynamics()
            for k in range(10000*sweepSize):                            # find total magnetisation
                self.dynamics()
                if (k%writeEvery == 0):
                    heatCapFile.write(str(self.computeTotalE())+",")
            heatCapFile.write("\n")
            print ("The simulation is {0:.2f}% completed.").format((i-10)*100./20.)
        heatCapFile.close()
        

    def createCanvas(self):
        self.plotFigure, self.ax = plt.subplots()
        self.plotFigure.suptitle("Ising Model Dynamics")                # creating axis and plots for plotting
        self.plotFigure.set_facecolor('black')
        self.ax.set_axis_bgcolor('0.05')
        self.ax.set_ylim(0,self._yDim+1)
        self.ax.set_xlim(0,self._xDim+1)
    
    def drawFrame(self):                                                # plots every arrow current position, arrows different colours depending +1/-1
        self.ax.set_xlabel('Total number of switches: '+str(self.totalSwitches)+
                           '\nTotal Up: '+str(self.noUp)+
                           ' Total Down: '+str(self.noDn)+
                           '\nTotal number of sweeps: '+str(self.noSweeps),color="w")
        
        self.ax.imshow(self.system,interpolation='nearest')
        if self.vis == 'F': self.ax.imshow(self.system,interpolation='nearest')
        elif self.vis == 'C':
            for i in xrange(self._xDim):
                for j in xrange(self._yDim):
                    val = self.system[i][j]
                    self.ax.arrow(i+1, j+1-0.25*val, 0, 0.5*val,           # plots cool looking arrows
                                  head_width=0.3, head_length=0.15, 
                                  fc=arrCol[val], ec=arrCol[val])

    def updateFrame(self,x):                                            # updates frame every xDim * yDim calculations
        updateSize = 10*self._xDim*self._yDim
        for i in range(updateSize):
            self.dynamics()
        self.noSweeps += updateSize/(self._xDim*self._yDim)
        self.ax.clear()
        self.drawFrame()                                                # plots frame again
	
    def animateSim(self):                                               # animates dynamics
        self.createCanvas()
        ani = animation.FuncAnimation(self.plotFigure, self.updateFrame,
                                      init_func=self.drawFrame,interval=1)
        plt.show()
        
    

class KawasakiDynamics(IsingModel):
    
    def setType(self): self.dynamicsType = "KawasakiDynamics"
    
    def probability(self,co1,co2):
        locDelE = self.computeDltaE(co1)+self.computeDltaE(co2)
        prob = np.exp(-locDelE/(self._kB*self.T))
        if self.NearNb:
            locDelE = self.computeDltaE(co1)+self.computeDltaE(co2)-self._J
            prob=np.exp(-locDelE/(self._kB*self.T))
        return prob

    def isNearNbour(self,co1,co2):
        distance = 0
        distance += abs(co1[0]-co2[0])
        distance += abs(co1[1]-co2[1])
        if distance ==1:
            self.NearNb = True
            return True
        else:
            self.NearNb = False
    
    def dynamics(self):
        rdCo1 = self.getRndCoords()
        rdCo2 = self.getRndCoords()
        self.randProb = rd.random()
        
        while True:
            if self.sysElmnt(rdCo1)==self.sysElmnt(rdCo2):
                rdCo1 = self.getRndCoords()
                rdCo2 = self.getRndCoords()
            else:
                break

        self.isNearNbour(rdCo1,rdCo2)
        probAccept = self.probability(rdCo1,rdCo2)
        if probAccept>1: probAccept = 1
        if probAccept >= self.randProb:
            self.switch(rdCo1)
            self.switch(rdCo2)
            #self.totalE +=- self.locDelE
        
	
class GlauberDynamics(IsingModel):
    """
    Subclass of Isingmodel superclass with glauber dynamic specific 
    methods to calculate probability of flipping acceptance.
    """
    def setType(self): self.dynamicsType = "GlauberDynamics"
    
    def probability(self,coOrd):
        locDelE = self.computeDltaE(coOrd)
        prob = np.exp(-locDelE / (self._kB*self.T))
        return prob
        
    def dynamics(self):
        rdCo1 = self.getRndCoords()
        self.randProb = rd.random()
        probAccept = self.probability(rdCo1)
        
        if probAccept>1: probAccept = 1
        
        if probAccept >= self.randProb:
            self.switch(rdCo1)
            
        
    def calcMagSusc(self):		
        self.system = np.ones((self._xDim,self._yDim))
        sweepSize = self._xDim*self._yDim                                # define how big a sweep is
        magSusFile = open("MagneticSusceptibilityGlauberDynamics.txt","w")
        magSusFile.write("Magnetic Susceptibility Measurements \n \n\
The size of the system:                     {}\n\
The number of sweeps to equilibriate:       300\n\
The number of measurements per temperature: 1000\n\n\
=============================================\n\
Temperature \nTotal Magnetisation Values\n\
=============================================\n\n".format(sweepSize))
		
        writeEvery = 10*sweepSize
        
        for i in range(10,30):                                          # loop through temps from 0.1 -> 3.0
            self.T=(i*0.1)
            magSusFile.write(str(self.T)+"\n")                          # writes T on top line
            
            for j in range(500*sweepSize):                              # equilibrate the system
                self.dynamics()
            for k in range(10000*sweepSize):                            # find total magnetisation
                self.dynamics()
                if (k%writeEvery == 0):
                    totalM = self.noUp - self.noDn
                    magSusFile.write(str(self.computeUpDn())+",")
            magSusFile.write("\n")
            print ("This simulation is {0:.2f}% completed.").format((i-10)*100./20.)
        magSusFile.close()
#_______________________________________________________________________


#=======================================================================

class FileAnalysis(object):
    def __init__(self,fileName):
        self.x    = []
        self.y    = []
        self.yErr = []
        self.sysSize = 2500
        
        self.loadFile(fileName)

        
    def loadFile(self,fName):
        fileStream = open(fName, "r")
        lnNo = 0

        for _ in xrange(11):
            next(fileStream)

        for line in fileStream:
            lnNo+=1
            if ((lnNo+1)%2==0): self.x.append(float(line.strip(" \n")))
            if (lnNo%2==0):
                specValues = [float(s) for s in line.strip(" \n")[0:-1].split(",")]
                self.y.append(self.computeFeature(specValues))
                self.yErr.append(self.computeBootstrapError(specValues))


    def computeFeature(self,specVals):
        pass


    def computeBootstrapError(self,sampVals):
        resampleFeatures = []
        for i in range(10):
            randomList = []
            for i in range(len(sampVals)):
                elmt = rd.randint(0,len(sampVals))
                randomList.append(sampVals[elmt])
            resampleFeatures.append(self.computeFeature(randomList))
            
        avResample = np.mean(resampleFeatures)
        avResample2 = np.mean(np.square(resampleFeatures))
        error = (avResample2 - avResample)**0.5
        return error

class HeatAnalysis(FileAnalysis):
    def computeFeature(self,specVals):
        specVals = np.array(specVals)
        avTotalE = np.average(specVals)
        avTotalEsq = np.average(np.square(specVals))
        heatC = (1./(self.sysSize*1*self.x[-1]**2.))*(avTotalEsq - avTotalE**2.)
        return heatC
        
class MagAnalysis(FileAnalysis):
    def computeFeature(self,specVals):
        specVals = np.array(specVals)
        avTotalM = np.average(abs(specVals))
        avTotalMsq = np.average(np.square(specVals))
        magSusc = (1./(self.sysSize*1*self.x[-1]))*(avTotalMsq - avTotalM**2.)
        return magSusc

def plot():
    CvGla = HeatAnalysis("HeatCapacityGlauberDynamics.txt")
    CvKaw = HeatAnalysis("HeatCapacityGlauberDynamics.txt")
    XmGla = MagAnalysis("MagneticSusceptibilityGlauberDynamics.txt") 
    
    plt.show()
    
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=False,sharey=False)
    ax1.errorbar(XmGla.x, XmGla.y, yerr = XmGla.yErr, fmt = '-o')
    ax2.errorbar(CvGla.x, CvGla.y, yerr = CvGla.yErr, fmt = '-o')
    ax3.errorbar(CvKaw.x, CvKaw.y, yerr = CvKaw.yErr, fmt = '-o')
    
    ax1.set_title("Magnetic Suseptibility Glauber Dynamics")
    ax3.set_xlabel("Temperature /K")
    ax2.set_title("Heat Capacity Glauber Dynamics")
    ax3.set_title("Heat Capacity Kawasaki Dynamics")
    plt.show()



#======================================================================#
typeError = "Please enter the type of dynamics you wish to use to model\
 the system as either 'GlauberDynamics' or 'KawasakiDynamics'."

sizeError = "Please give the desired size of the system in the format\
xDimension,yDimension. For example: '50x50'."

tempError = "Please provide a temperature as your final argument in the\
form of a decimal number. For example: '2.3'."
#======================================================================#


def debug():
	
    if (sys.argv[1]=="KawasakiDynamics" or 
        sys.argv[1]=="GlauberDynamics"): dynamicsType = globals()[sys.argv[1]]
    else:
		print "\n"+typeError+"\n"
		sys.exit()

    if len(sys.argv)==4:
        dimenL = sys.argv[2].split("x")
        if len(dimenL)!=2: 
            print "\n"+sizeError+"\n"
            sys.exit()
        dimenL = map(int, dimenL)
        
        try: temp = float(sys.argv[3])
        except: print tempError
        
        model0 = dynamicsType(0, size=np.array([dimenL[0],dimenL[1]]), tmp = temp)
        
    elif len(sys.argv)>4: print "Please only provide dynamics type, system dimensions, and temperature (3 args)"
    
    else: print sizeError+"\n\n"+tempError
    
    model0.calcMagSusc()
    
def doMeasure():
    model0 = GlauberDynamics(0)
    model1 = GlauberDynamics(1)
    model2 = KawasakiDynamics(2)
    
    model2.calcHeatCap()
    model1.calcHeatCap()
    model0.calcMagSusc()


def temp():
    T  = raw_input("\nHow hot you want this system to be?\n\n")
    if '.' in T and len(T)>2:
        try:    return float(T)
        except: temp()
    else: temp()

def dimens():
    dimenSize = raw_input("\nPlease give desired system size in format x_dim x y_dim (eg. '50x50'): \n\n")
    dimenL = dimenSize.split("x")
    try: dimenL = map(int, dimenL)
    except: dimens()
    if len(dimenL) == 2: return np.array([dimenL[0],dimenL[1]])
    
def dynams():
    dynoType = raw_input("\nPlease indicate whether you wish to model Glauber ('G') or Kawasaki ('K') dynamics: \n\n")
    if (dynoType == 'G'): return 'G'
    elif (dynoType == 'K'): return 'K'
    else: dynams()
    
def fastOrPretty():
    visType = raw_input("\nWould you like the visualisation to be fast ('F'), or cool looking ('C')? \n\n")
    if (visType == 'F'): return 'F'
    elif (visType == 'C'): return 'C'
    else: fastOrPretty()
    
def animQuestions():
    dynamType = dynams()
    systemSize = dimens()
    temperature = temp()
    visualType = fastOrPretty()
    typeDict = dict([('G', 'GlauberDynamics'), ('K', 'KawasakiDynamics')])
    dynamicsType = globals()[typeDict[dynamType]]
    model0 = dynamicsType(0, size = systemSize, tmp = temperature, vis = visualType)
    model0.animateSim()
    
def mainQuestions():
    hello = raw_input("\nPlease indicate whether you wish to animate an Ising model ('anim') or plot previously calculated results ('plot').\n\n")
    if (hello=="anim"): animQuestions()
    elif (hello=="plot"): plot()
    else: mainQuestions()
    
mainQuestions()
#debug()
#doMeasure()
