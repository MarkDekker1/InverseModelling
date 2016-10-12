# Inverse Modelling using Tracer Class

from Class_Tracer import *

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

# construct large matrix, relates x_0 = [c_i,E_i] = x_n = [c_n,E_i]
F = np.zeros((2*nx,2*nx))
F[:nx,:nx] = m1.Mtot.toarray()
F[:nx,nx:] = np.diag(np.ones(nx)*m1.P['dt'])
F[nx:,nx:] = np.diag(np.ones(nx))

# Define indices of timeseries to take
#indices = [10,30,50,80] # station locations
#indices = np.array([20,70,90]) * nx//100

numindices = len(indices)

Khatlarge = np.zeros((numindices*nt,nx))
yinlarge = np.zeros(numindices*nt)

for numid in range(numindices):
    print('Processing timeseries %i ' % (numid+1))
    i = indices[numid]
    yinlarge[numid*nt:(numid+1)*nt] = m1.results[:,i]
    
    K = np.zeros((nt,2*nx))
    for ni in range(nt):
        #print(ni)
        K[ni] = matrix_power(F,ni)[i]

    Khat = K[:,100:]
    
    Khatlarge[numid*nt:(numid+1)*nt,:] = Khat

# add additive and multiplicative noise
yinlarge = yinlarge * (1 + noisemult * np.random.uniform(low=-1,high=1,size=len(yinlarge))) + noiseadd * np.random.uniform(low=-1,high=1,size=len(yinlarge))
# Compute best estimate
xa = np.zeros(nx) # all zeros, no prior knowledge

#Sa = np.diag(sigmaxa*np.ones(2*nx))
Sa = np.diag(sigmaxa*np.ones(nx))
Se = np.diag(sigmaxe*np.ones(nt*numindices))

print('Computing best estimate')
# construct G matrix (Jacob, eq. 5.9)
G = np.matmul(np.matmul(inv(np.matmul(np.matmul(Khatlarge.transpose(),inv(Se)),Khatlarge)+inv(Sa)),Khatlarge.transpose()),inv(Se))
# compute initial vector based on yin, K, xa (Jacob, eq. 5.7)
x = xa + np.matmul(G,yinlarge-np.matmul(Khatlarge,xa))

# Error covariance of initial vector (Jacob, eq. 5.10)
Shat = inv(np.matmul(np.matmul(Khatlarge.transpose(),inv(Se)),Khatlarge) + inv(Sa))

# RMS deviation between actual and recovered sources
rmsdev = ( np.mean((x-m1.P['E'])**2)  )**0.5