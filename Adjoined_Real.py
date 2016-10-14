#%% ------------------------------------------------------
# Preambule
# ------------------------------------------------------

from numpy import *
from Class_Tracer import *

xmax=100
Dx=1
xlen=np.int(xmax/Dx)
tmax=1000
Dt=0.1
tlen=np.int(tmax/Dt)
xvec=np.zeros(100)
k=0.1
u0 = 5
nx=xmax/Dx
nt=tmax/Dt

# ------------------------------------------------------
# Define truth
# ------------------------------------------------------

E_true=np.zeros(100)
E0=1
for j in range(0,xlen):
    x=j*Dx
    xvec[j]=x
    if x==1 or x==10 or x==50:
        E_true[j]=E0
C0_true = np.zeros(100)

# ------------------------------------------------------
# Observations
# ------------------------------------------------------
        
Parameter_initial = {
    'xmax':xmax,
    'dx':Dx,
    'tmax':tmax,
    'dt':Dt,
    'u0':u0,
    'k':k,
    'E':E_true
    }

m1 = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
m1.integrateModel()
Obs_true=m1.results

# ------------------------------------------------------
# Transport matrix F
# ------------------------------------------------------

F = np.zeros((200,200))  
F[:100,:100] = m1.Mtot
F[:100,100:] = np.diag(np.ones(100)*m1.P['dt'])
F[100:,100:] = np.diag(np.ones(100))
Transport = m1.Mtot
TransEmis = F

# ------------------------------------------------------
# First guess
# ------------------------------------------------------

E_guess=np.zeros(100)
E0=1
for j in range(0,xlen):
    x=j*Dx
    xvec[j]=x
    if x==2 or x==12 or x==80:
        E_guess[j]=E0
C0_guess = zeros(100)

# ------------------------------------------------------
# Guess run
# ------------------------------------------------------
        
Parameter_initial = {
    'xmax':xmax,
    'dx':Dx,
    'tmax':tmax,
    'dt':Dt,
    'u0':u0,
    'k':k,
    'E':E_guess
    }

m2 = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
m2.integrateModel()
Obs_guess=m2.results
Cost_initial = sum((array(Obs_guess)-array(Obs_true))**2)

# ------------------------------------------------------
# Adjoined model
# ------------------------------------------------------

sigmaxa = 0.001      # prior (0.001)
sigmaxe = 2e-8      # observations (2e-8)
Sa = np.diag(sigmaxa*np.ones(nx))
Se = np.diag(sigmaxe*np.ones(nt))
forcing = array(Obs_guess)-array(Obs_true)  # (Hx-y)

timevec=range(0,int(tmax/Dt))
C_adjoined = zeros(100)
E_adjoined = zeros(100)
for times in timevec[::-1]:
    C_adjoined = C_adjoined + forcing[times]
    C_adjoined = matmul(np.transpose(Transport),C_adjoined)
    E_adjoined = E_adjoined + C_adjoined*Dt
    #C_adjoined[100:] = C_adjoined[100:] + C_adjoined[:100]
    #E_adjoined = E_adjoined + C_adjoined
#print('Guess dJ/dC:',2*C_adjoined, 2*E_adjoined)

# ------------------------------------------------------
# Test derivative for E[TestElement]
# ------------------------------------------------------

alpha=0.0001
TestElement=0
E_test=E_guess
E_test[TestElement]=E_guess[TestElement]+alpha
Parameter_initial = {
    'xmax':xmax,
    'dx':Dx,
    'tmax':tmax,
    'dt':Dt,
    'u0':u0,
    'k':k,
    'E':E_test
    }

m3 = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
m3.integrateModel()
Obs_test=m3.results
Cost_test = sum((array(Obs_test)-array(Obs_true))**2)
print('Old cost function',Cost_initial)
print('New cost function:',Cost_test)
print('Calculated-derivative:',2*E_adjoined[TestElement])
print('Test-derivative:', (Cost_test-Cost_initial)/alpha)

# ------------------------------------------------------
# Implement DE (=Delta E)
# ------------------------------------------------------

Derivative=E_adjoined
DE=zeros(100)-0.00001
E_new=E_guess+DE*Derivative
Parameter_initial = {
    'xmax':xmax,
    'dx':Dx,
    'tmax':tmax,
    'dt':Dt,
    'u0':u0,
    'k':k,
    'E':E_new
    }

m4 = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
m4.integrateModel()
Obs_new=m4.results
Cost_new = sum((Obs_new-Obs_true)**2)
print('Old cost function',Cost_initial)
print('New cost function:',Cost_new)

# ------------------------------------------------------
# Plot
# ------------------------------------------------------

matplotlib.style.use('ggplot')
fig=plt.figure(num=None, figsize=(7,4),dpi=150, facecolor='w', edgecolor='k') # little: 5,3, large: 9,3
plt.plot(Obs_true[999],'k',label='True',linewidth=3)
plt.plot(Obs_guess[999],label='First guess',linewidth=3)
plt.plot(Obs_new[999],label='New',linewidth=3)
plt.xlabel('Distance x',fontsize=15)
plt.ylabel('End concentration',fontsize=15)
plt.xlim([0,100])
plt.ylim([0,1])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(loc='best',fontsize=12)
fig.tight_layout()
plt.show()

matplotlib.style.use('ggplot')
fig=plt.figure(num=None, figsize=(7,3),dpi=150, facecolor='w', edgecolor='k') # little: 5,3, large: 9,3
plt.plot(E_true,'k',label='True',linewidth=3)
plt.plot(E_guess,label='First guess',linewidth=3)
plt.plot(E_new,label='New',linewidth=3)
plt.xlabel('Distance x',fontsize=15)
plt.ylabel('Emission strength',fontsize=15)
plt.xlim([0,100])
plt.ylim([0,1])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(loc='best',fontsize=12)
fig.tight_layout()
plt.show()
#%%
E_guess=np.zeros(100)
E0=1
for j in range(0,xlen):
    x=j*Dx
    xvec[j]=x
    if x==2 or x==12 or x==80:
        E_guess[j]=E0
        
Parameter_initial = {
    'xmax':xmax,
    'dx':Dx,
    'tmax':tmax,
    'dt':Dt,
    'u0':u0,
    'k':k,
    'E':E_guess
    }

m2 = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
m2.integrateModel()
Obs_guess=m2.results
forcing = array(Obs_guess)-array(Obs_true)  # (Hx-y)

timevec=range(0,int(tmax/Dt))
C_adjoined = zeros(100)
E_adjoined = zeros(100)
for times in timevec[::-1]:
    C_adjoined = C_adjoined + forcing[times]
    C_adjoined = matmul(np.transpose(Transport),C_adjoined)
    E_adjoined = E_adjoined + C_adjoined*Dt

#%%
sigmaxa = 0.01      # prior (0.001)
sigmaxe = 0.01      # observations (2e-8)
Sa = np.diag(sigmaxa*np.ones(nx))
Se = np.diag(sigmaxe*np.ones(nt))
Sai = np.diag((1/sigmaxa)*np.ones(nx))
Sei = np.diag((1/sigmaxa)*np.ones(nt))
#Sai= np.linalg.inv(Sa)
#Sei= np.linalg.inv(Se)

#%%
# From calculation, it is expected that the local minimum occurs at x=9/4
E_guess=np.zeros(100)
E0=1
for j in range(0,xlen):
    x=j*Dx
    xvec[j]=x
    if x==2 or x==12 or x==80:
        E_guess[j]=E0
E_guess=np.zeros(100)
        
matplotlib.style.use('ggplot')
fig=plt.figure(num=None, figsize=(9,5),dpi=150, facecolor='w', edgecolor='k') # little: 5,3, large: 9,3
plt.plot(E_true,'k',label='True',linewidth=3)
plt.plot(E_guess,label='First guess',linewidth=3)

x_old = np.zeros(100) # The value does not matter as long as abs(x_new - x_old) > precision
x_new = E_guess # The algorithm starts at x=6
x_prior = E_guess
print(max(abs(x_new - x_old)))

gamma = 0.00000001 # step size
precision = 0.000000001



def df(a,b):
    Parameter_initial = {
        'xmax':xmax,
        'dx':Dx,
        'tmax':tmax,
        'dt':Dt,
        'u0':u0,
        'k':k,
        'E':a
        }
    
    m2 = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
    m2.integrateModel()
    Obs_guess=m2.results
    forcing = matmul(Sei,array(Obs_guess)-array(Obs_true))  # (Hx-y)
    
    timevec=range(0,int(tmax/Dt))
    C_adjoined = zeros(100)
    E_adjoined = zeros(100)
    for times in timevec[::-1]:
        C_adjoined = C_adjoined + forcing[times]
        C_adjoined = matmul(np.transpose(Transport),C_adjoined)
        E_adjoined = E_adjoined + C_adjoined*Dt
    
    derivative = 2*matmul(Sai,a-b)+ 2*E_adjoined    
    
    return derivative

#while max(abs(x_new - x_old)) > precision:
for j in range(0,50):
    x_old = x_new
    x_new = x_new - gamma * df(x_old,x_prior)
    print(mean(abs(x_new - x_old)))
    
#
plt.plot(x_new,label='New',linewidth=3)
plt.xlabel('Distance x',fontsize=15)
plt.ylabel('Emission strength',fontsize=15)
plt.xlim([0,100])
plt.ylim([0,2])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(loc='best',fontsize=12)
fig.tight_layout()
plt.show()
#print("The local minimum occurs at ", +x_new)

#%% 
# ------------------------------------------------------
# Looped
# ------------------------------------------------------

Parameter_initial = {'xmax':xmax,'dx':Dx,'tmax':tmax,'dt':Dt,'u0':u0,'k':k,'E':E_true}
m1 = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
m1.integrateModel()
Obs_true=m1.results
E_guess=np.zeros(100)
E0=1
for j in range(0,xlen):
    x=j*Dx
    xvec[j]=x
    if x==2 or x==12 or x==80:
        E_guess[j]=E0
E_guess=np.zeros(100)

matplotlib.style.use('ggplot')
fig=plt.figure(num=None, figsize=(7,7),dpi=150, facecolor='w', edgecolor='k') # little: 5,3, large: 9,3
plt.plot(E_true,'k',label='True',linewidth=3)

for q in range(1,50):
    
    Parameter_initial = {'xmax':xmax,'dx':Dx,'tmax':tmax,'dt':Dt,'u0':u0,'k':k,'E':E_guess}
    m2 = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
    m2.integrateModel()
    Obs_guess=m2.results
    
    
    forcing = array(Obs_guess)-array(Obs_true)
    timevec=range(0,int(tmax/Dt))
    C_adjoined = zeros(100)
    E_adjoined = zeros(100)
    for times in timevec[::-1]:
        C_adjoined = C_adjoined + forcing[times]
        C_adjoined = matmul(np.transpose(Transport),C_adjoined)
        E_adjoined = E_adjoined + C_adjoined*Dt
    Derivative=E_adjoined
    DE=zeros(100)-0.00001
    E_new=E_guess+DE*Derivative
    print(q)
    
    E_guess=E_new



    plt.plot(E_new,label='First guess',linewidth=1)
plt.xlabel('Distance x',fontsize=15)
plt.ylabel('Emission strength',fontsize=15)
plt.xlim([0,100])
plt.ylim([0,1])
plt.tick_params(axis='both', which='major', labelsize=15)
#plt.legend(loc='best',fontsize=12)
fig.tight_layout()
plt.show()