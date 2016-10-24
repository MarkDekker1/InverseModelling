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
Dx              =   xmax / xmax
Dt              =   gamma * Dx / u0
E0              =   1
sources         =   [10,30,60,90]
#sources         =   np.random.randint(low=0,high=xmax,size=3)
sources_guess   =   [100000]               #for determining prior
stations        =   np.arange(0,xmax,1)     #all points have stations
#stations        =   [20,40,60,80]
stations        =   np.random.randint(low=0,high=xmax,size=1)
stations        =   [26,46,61,73,92,95]
#stations        =   np.arange(0,100,1)
sigmaxa         =   0.5
sigmaxe         =   0.4
accuracy        =   1e-5
noisemult       =   0
noiseadd        =   0.004
Preconditioning =   0
Rerunning       =   5
Offdiags        =   0
BFGS            =   1
Print           =   0
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

RSME=mean(sqrt(sum((m.Obs_final-m.Obs_true)**2)/1111.))
print('RMSE is ',RSME)


maximum=max(m.E_final)
print('Highest peak is ',maximum)

a=[]
for i in range(0,99):
    a.append(abs(m.E_final[i+1]-E_true[i]))
largestdev=max(abs(m.E_final-E_true))
print('Largest deviation is ',largestdev)

#%%
Save_1=m.E_final
#%%
Save_4=m.E_final
#%%
Save_6=m.E_final
#%%
Finals=[Save_1,Save_4,Save_6]
stations_vec=[[33],[9,26,43,83],[26,46,61,72,92,95]]
fontsize1=20
matplotlib.style.use('ggplot')
plt.style.use('seaborn-white')
fig=plt.figure(num=None, figsize=(8,5),dpi=600, facecolor='w', edgecolor='k') 

for i in range(0,3):
#for i in Best:
#for i in Best:
    a=plt.plot(xvec+1,np.transpose(Finals[i]),label='Determined with preco',linewidth=4)
    plt.scatter(stations_vec[i],np.zeros(len(stations_vec[i]))+0.02,s=100,c=a[0].get_color(),alpha=1,zorder=15,edgecolor='k',linewidth=2,label='Measuring Stations')

plt.plot(E_true,'dimgray',label='Actual',linewidth=2,zorder=-1)
#plt.plot([0,0],[0.5,0.5],'k',linewidth=5,zorder=15)
plt.xlabel('Distance x',fontsize=fontsize1)
plt.ylabel('Emission strength',fontsize=fontsize1)
plt.tick_params(axis='both', which='major', labelsize=fontsize1)
plt.xlim([0,xmax])
plt.ylim([-0.2,1])
#plt.legend(loc='best')
fig.tight_layout()
plt.show()