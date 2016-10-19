# Inverse Modelling using Tracer Class

from Class_Tracer_v2 import *

from numpy.linalg import matrix_power,inv
import scipy.linalg as la


# FORWARD PARAMETERS
#Constants
xmax    =   100 # 40000e3
u0      =   5
tmax    =   10* xmax / u0
k       =   0.1 # 1.93e-7 # based on 60 day lifetime of CO
gamma   =   0.9
Dx      =   xmax / 100
Dt      =   gamma * Dx / u0 #0.1
E0      =   1 # source strength
sources = [10,50,60] # source locations as percentage of xmax

# INVERSE PARAMETERS
stations    = [20,70,90] # measurement stations as percentage of xmax
# error estimates 
sigmaxa     = 0.001 #10        #e-9 #20#0.00001 # error in the prior
sigmaxe     = 2e-5  #0.0000001 #    #0.00000000001 # error in the observations
noiseadd    = 0.01 # additive noise amplitude on measurements in concentration units
noisemult   = 0.01 # multiplicative noise amplitude on measurements (fraction)
# ======================================
# DERIVED PARAMETERS
nx      =   np.int(xmax/Dx)
nt      =   np.int(tmax/Dt)
xvec    =   np.arange(0,nx,Dx) #np.zeros(100)

#Set up source array E
E   =   np.zeros(nx)
source_ids = (np.array(sources) * nx//100).astype(int)
#source_mag = [nx*k] # source strengths
source_mag = [1/Dx] # source strengths
E[source_ids] = source_mag

#indices = np.array([20,70,90]) * nx//100 # measurement stations
indices = (np.array(stations) * nx//100).astype(int) # measurement stations

# ======================================
# FORWARD
# ======================================

# Forward step
Parameter_initial = {
    'xmax':xmax,
    'dx':Dx,
    'tmax':tmax,
    'dt':Dt,
    'u0':u0,
    'k':k,
    'E':E
    }

# Forward integration
m1 = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
m1.integrateModel()


# ======================================
# INVERSE
# ======================================

inverseParams = {
	'stations':stations,
	'sigmaxa':sigmaxa,
	'sigmaxe':sigmaxe,
	'noiseadd':noiseadd,
	'noisemult':noisemult
	}

m1.inverseSetup(inverseParams)
m1.inverseKmatrix()
m1.inverseGmatrix()

m1.inverseSolution()


fig,ax = plt.subplots(1)
ax.plot(m1.x,m1.P['E'])
ax.plot(m1.x,m1.Einv)

print(m1.rmsdev)