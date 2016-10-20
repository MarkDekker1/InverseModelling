# ------------------------------------------------------
# Preambule
# ------------------------------------------------------

from numpy import *
from Class_Tracer_v2 import *
from scipy.optimize import *

# ------------------------------------------------------
# Define parameters
# ------------------------------------------------------

xmax        =   100
u0          =   5
tmax        =   10* xmax / u0
k           =   0.1
gamma       =   0.9
Dx          =   xmax / xmax
Dt          =   gamma * Dx / u0
Dt          =   0.1
E0          =   1
sources     =   [1,10,50]#,50]
sources_guess=  [30000]
stations    =   np.arange(0,xmax,1)#[5,10,20]#,40]#,60,80,90]
stations    =   [2,5,9,15,20]#,28,32,40,60,80]
stations    =   [20,40,60,80]
#stations    =   np.random.random_integers(low=0,high=100,size=6)
nx          =   np.int(xmax/Dx)
nt          =   np.int(tmax/Dt)
xvec        =   np.arange(0,xmax,Dx)
sigmaxa     =   10
sigmaxe     =   0.001
accuracy        =   1e-5
noisemult   =   0#0.03
noiseadd    =   0.003
Preconditioning = 0
Rerunning   =   3

# ------------------------------------------------------
# Emission vectors and error covariance matrices
# ------------------------------------------------------

E_prior=np.zeros(nx)
E_true=np.zeros(nx)
E0=1
for j in range(0,nx):
    x=np.int(j*Dx)
    xvec[j]=x
    if len(where(np.array(sources_guess)==x)[0])>0:
        E_prior[j]=E0
    if len(where(np.array(sources)==x)[0])>0:
        E_true[j]=E0
        
Parameters_true = {'xmax':xmax,'dx':Dx,'tmax':tmax,'dt':Dt,'u0':u0,'k':k,'E':E_true}
Parameters_prior = {'xmax':xmax,'dx':Dx,'tmax':tmax,'dt':Dt,'u0':u0,'k':k,'E':E_prior}

Sa = np.diag(sigmaxa*np.ones(nx))
Se = np.diag(sigmaxe*np.ones(nt))
Sai = np.diag((1/sigmaxa)*np.ones(nx))
Sei = np.diag((1/sigmaxe)*np.ones(nt))

# ------------------------------------------------------
# Define function adjoined model and cost function
# ------------------------------------------------------

def Adjoined(x,x_prior):
    Parameters_iteration = {'xmax':xmax,'dx':Dx,'tmax':tmax,'dt':Dt,'u0':u0,'k':k,'E':x}
    m = TracerModel(Parameters_iteration,method='Upwind',initialvalue=0)
    m.integrateModel()
    Obs_iteration=m.results[:,stations]
    forcing = matmul(Sei,array(Obs_iteration)-array(Obs_true))  # (Hx-y)
    timevec=range(0,nt)
    C_adjoined = np.zeros(nx)
    E_adjoined = np.zeros(nx)
    for times in timevec[::-1]:
        C_adjoined[stations] = C_adjoined[stations] + forcing[times]
        C_adjoined = matmul(np.transpose(Transport),C_adjoined)
        E_adjoined = E_adjoined + C_adjoined*Dt
    derivative = 2*matmul(Sai,x-x_prior)+ 2*E_adjoined   
    return derivative
    
def Cost(x):
    Parameters = {'xmax':xmax,'dx':Dx,'tmax':tmax,'dt':Dt,'u0':u0,'k':k,'E':x}
    m = TracerModel(Parameters,method='Upwind',initialvalue=0)
    m.integrateModel()
    Obs=m.results[:,stations]
    Cost=1/sigmaxe*sum((array(Obs_true)-array(Obs))**2)+1/sigmaxa*sum((array(x)-array(E_prior))**2)
    return Cost

# ------------------------------------------------------
# Forward run and define F matrix
# ------------------------------------------------------

m_true = TracerModel(Parameters_true,method='Upwind',initialvalue=0)
m_true.integrateModel()
Obs_true_raw=m_true.results[:,stations]
Obs_true=[]
for i in range(0,len(Obs_true_raw)):
    Obs_true.append( Obs_true_raw[i]*(1+noisemult*np.random.uniform(low=-1,high=1,size=(len(Obs_true_raw[i]))))+noiseadd*np.random.uniform(low=-1,high=1,size=(len(Obs_true_raw[i]))))
Obs_true=np.array(Obs_true)

F = np.zeros((2*nx,2*nx))  
F[:nx,:nx] = m_true.Mtot2
F[:nx,nx:] = np.diag(np.ones(nx)*m_true.P['dt'])
F[nx:,nx:] = np.diag(np.ones(nx))
Transport = m_true.Mtot2
TransEmis = F
    
# ------------------------------------------------------
# Prior run
# ------------------------------------------------------
    
Cost_prior = Cost(E_prior)
Derivative = Adjoined(E_prior,E_prior)

# ------------------------------------------------------
# Test derivative adjoined model for Qth element of emission vector E
# ------------------------------------------------------

alpha=0.00001
Q=15
E_test=np.zeros(nx)
E0=1
for j in range(0,nx):
    x=np.int(j*Dx)
    xvec[j]=x
    if len(where(np.array(sources_guess)==x)[0])>0:
        E_test[j]=E0
E_test[Q]=E_test[Q]+alpha
Cost_test = Cost(E_test)


# ------------------------------------------------------
# Implement DE (=Delta E)
# ------------------------------------------------------

DE=np.zeros(nx)-0.000000001
E_new=np.zeros(nx)
E0=1
for j in range(0,nx):
    x=np.int(j*Dx)
    xvec[j]=x
    if len(where(np.array(sources_guess)==x)[0])>0:
        E_new[j]=E0
E_new=E_new+DE*Derivative
Cost_new = Cost(E_new)

# ------------------------------------------------------
# Print test results and the like
# ------------------------------------------------------

print('================================================')
print('--- Initial conditions and tests ---')
print('Prior cost function',Cost_prior)
print('Derivative by calculation:',Derivative[Q])
print('Derivative by test:', (Cost_test-Cost_prior)/alpha)
print('Cost function one time',Cost_new)
print('Cost function change',Cost_new-Cost_prior)

# ------------------------------------------------------
# Adjoined modelling
# ------------------------------------------------------

print('================================================')
print('--- Adjoined modelling ---')

def Adjoined_priorint(x):
    '''
    adjoined model with given E_prior so that it has only 1 element
    '''
    Parameters_iteration = {'xmax':xmax,'dx':Dx,'tmax':tmax,'dt':Dt,'u0':u0,'k':k,'E':x}
    m = TracerModel(Parameters_iteration,method='Upwind',initialvalue=0)
    m.integrateModel()
    Obs_iteration=m.results[:,stations]
    forcing = matmul(Sei,array(Obs_iteration)-array(Obs_true))  # (Hx-y)
    C_adjoined = np.zeros(nx)
    E_adjoined = np.zeros(nx)
    timevec=range(0,nt)
    for times in timevec[::-1]:
        C_adjoined[stations] = C_adjoined[stations] + forcing[times]
        C_adjoined = matmul(np.transpose(Transport),C_adjoined)
        E_adjoined = E_adjoined + C_adjoined*Dt
    derivative = 2*matmul(Sai,x-E_prior)+ 2*E_adjoined
    deriv=dot(L_adj,derivative)
    print ('Cost function', Cost(x), 'Squared gradient',dot(deriv,deriv))
    return deriv

def state_to_precon( L_inv, L_adj, state, state_apri, deriv ):
    '''
    Same function used in example code for preconditioning
    '''
    pstate = dot(L_inv,array(state)-array(state_apri))
    pderiv = dot(L_adj,array(deriv))
    return pstate, pderiv

def precon_to_state( L_precon, vals, state_apri ):
    '''
    Same function used in example code for preconditioning
    '''
    state = dot(L_precon, vals) + array(state_apri)
    return state
    
    
b=Sa
L_preco = sqrt(b)
L_adj   = transpose(L_preco)
L_inv   = linalg.inv(L_preco)
vals    = Obs_true
E_final=E_prior

if Preconditioning==1:
    for i in range(0,Rerunning):
        pstate, pderiv = state_to_precon(L_inv, L_adj, E_final, E_prior, Adjoined_priorint(E_final))
        state_opt=optimize.fmin_bfgs(Cost,pstate,Adjoined_priorint,gtol=accuracy,disp=0)
        E_final = precon_to_state( L_preco, state_opt, E_prior )
elif Preconditioning==0:
    for i in range(0,Rerunning):
        E_final=optimize.fmin_bfgs(Cost,E_final,Adjoined_priorint,gtol=accuracy,disp=0)

print('--- Adjoined modelling completed ---')
print('================================================')
# ------------------------------------------------------
# Plots of emissions and one time series
# ------------------------------------------------------

matplotlib.style.use('ggplot')
fig=plt.figure(num=None, figsize=(7,3),dpi=150, facecolor='w', edgecolor='k') # little: 5,3, large: 9,3
plt.scatter(stations,np.zeros(len(stations))+0.03,s=100,c='orange',alpha=1,zorder=15,edgecolor='k',linewidth=2,label='Measuring Stations')
plt.plot(E_prior,label='Prior',linewidth=4)
plt.plot(E_true,'dimgray',label='Actual',linewidth=2)
plt.plot(E_final,label='Determined with preco',linewidth=4)
plt.xlabel('Distance x',fontsize=15)
plt.ylabel('Emission strength',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlim([0,xmax])
plt.ylim([-0.2,1])
plt.legend(fontsize=9)
fig.tight_layout()
plt.show()

Parameters_iteration = {'xmax':xmax,'dx':Dx,'tmax':tmax,'dt':Dt,'u0':u0,'k':k,'E':E_final}
m = TracerModel(Parameters_iteration,method='Upwind',initialvalue=0)
m.integrateModel()
Obs_final=m.results[:,stations]
Station = 0

matplotlib.style.use('ggplot')
fig=plt.figure(num=None, figsize=(7,3),dpi=150, facecolor='w', edgecolor='k') # little: 5,3, large: 9,3
plt.plot(Obs_true_raw[:,Station],'dimgray',label='Actual',linewidth=2)
plt.plot(Obs_true[:,Station],'orange',label='Actual measured',linewidth=2)
plt.plot(Obs_final[:,Station],label='Determined',linewidth=4)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Concentration',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlim([0,tmax])
plt.legend(fontsize=9,loc='best')
fig.tight_layout()
plt.show()