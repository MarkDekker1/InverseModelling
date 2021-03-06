# ------------------------------------------------------
# Define parameters
# ------------------------------------------------------

import numpy as np
import time as T

xmax            =   100
u0              =   5
tmax            =   10* xmax / u0
k               =   0.1
gamma           =   0.9
Dx              =   xmax / 100
Dt              =   gamma * Dx / u0
E0              =   1
sources         =   [1,10,50]
#sources         =   np.random.randint(low=0,high=xmax,size=3)
sources_guess   =   [100000]               #for determining prior
stations        =   np.arange(0,xmax,1)     #all points have stations
#stations        =   [20,40,60,80]
stations        =   np.random.randint(low=0,high=xmax,size=1)
#stations        =   [19]
#stations        =   np.arange(0,100,1)
sigmaxa         =   2
sigmaxe         =   0.004
accuracy        =   1e-5
noisemult       =   0
noiseadd        =   0.004
Preconditioning =   0
Rerunning       =   1
Offdiags        =   0
BFGS            =   1
Sa_vec          =   np.zeros(np.int(xmax/Dx))+sigmaxa#/10.
#for i in range(0,13):
#    Sa_vec[i]=sigmaxa
#for j in range(45,55):
#    Sa_vec[i]=sigmaxa
#for i in [0,10,50]:
#    Sa_vec[i]=sigmaxa
for i in range(0,25):
    Sa_vec[i]=sigmaxa
for i in range(40,70):
    Sa_vec[i]=sigmaxa
        
# ------------------------------------------------------
# Emission vectors
# ------------------------------------------------------

nx          =   np.int(xmax/Dx)
nt          =   np.int(tmax/Dt)
xvec        =   np.arange(0,xmax,Dx)
E_prior     =   np.zeros(nx)
E_true      =   np.zeros(nx)
for j in range(0,nx):
    x       =   j*Dx
    xvec[j] =   x
    if len(np.where(np.array(sources_guess)==x)[0])>0:
        E_prior[j]=E0
    if len(np.where(np.array(sources)==x)[0])>0:
        E_true[j]=E0
        
# ------------------------------------------------------
# Define dictionary for class
# ------------------------------------------------------
        
Parameters = {
    'xmax':xmax,
    'dx':Dx,
    'tmax':tmax,
    'dt':Dt,
    'u0':u0,
    'k':k,
    'E_prior':E_prior,
    'E_true':E_true,
    'stations':stations,
    'sigmaxa':sigmaxa,
    'sigmaxe':sigmaxe,
    'noisemult':noisemult,
    'noiseadd':noiseadd,
    'precon':Preconditioning,
    'rerunning':Rerunning,
    'accuracy':accuracy,
    'Offdiags':Offdiags,
    'BFGS':BFGS,
    'Sa_vec':Sa_vec
    }

# ------------------------------------------------------
# Testing the adjoint model
# ------------------------------------------------------

m = AdjointModel(Parameters,method='Upwind',initialvalue=0)

# ------------------------------------------------------
# Gaining results
# ------------------------------------------------------

#m.TestAdjoint(0.000001,1)
start_time = T.time()
m.AdjointModelling()
print('Total time required: %.2f seconds' % (T.time() - start_time))

# ------------------------------------------------------
# Plotting results
# ------------------------------------------------------

m.Plots(0)