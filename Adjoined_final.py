# ------------------------------------------------------
# Preambule
# ------------------------------------------------------

from numpy import *
from Class_Tracer import *

# ------------------------------------------------------
# Define parameters
# ------------------------------------------------------

xmax        =   100
u0          =   5
tmax        =   10* xmax / u0
k           =   0.1
gamma       =   0.9
Dx          =   xmax / 100
Dt          =   gamma * Dx / u0
Dt          =   0.1
E0          =   1
sources     =   [1,10,50]
sources_guess=  [10000]
stations    =   [5,13,43,60]
nx          =   np.int(xmax/Dx)
nt          =   np.int(tmax/Dt)
xvec        =   np.arange(0,xmax,Dx)
sigmaxa     =   0.01
sigmaxe     =   2e-5
epsilon     =   0.000000001#*sigmaxa*sigmaxe#0.000000000001
J           =   50 # amount of adjoined iterations

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
# Define function adjoined model
# ------------------------------------------------------

def Adjoined(x,x_prior):
    Parameters_iteration = {'xmax':xmax,'dx':Dx,'tmax':tmax,'dt':Dt,'u0':u0,'k':k,'E':x}
    m = TracerModel(Parameters_iteration,method='Upwind',initialvalue=0)
    m.integrateModel()
    Obs_iteration=m.results[:,stations]
    forcing = matmul(Sei,array(Obs_iteration)-array(Obs_true))  # (Hx-y)
    timevec=range(0,nt)
    C_adjoined = zeros(nx)
    E_adjoined = zeros(nx)
    for times in timevec[::-1]:
        C_adjoined[stations] = C_adjoined[stations] + forcing[times]
        C_adjoined = matmul(np.transpose(Transport),C_adjoined)
        E_adjoined = E_adjoined + C_adjoined*Dt
    derivative = 2*matmul(Sai,x-x_prior)+ 2*E_adjoined   
    return derivative

# ------------------------------------------------------
# Forward run and define F matrix
# ------------------------------------------------------

m_true = TracerModel(Parameters_true,method='Upwind',initialvalue=0)
m_true.integrateModel()
Obs_true=m_true.results[:,stations]

F = np.zeros((2*nx,2*nx))  
F[:nx,:nx] = m_true.Mtot
F[:nx,nx:] = np.diag(np.ones(nx)*m_true.P['dt'])
F[nx:,nx:] = np.diag(np.ones(nx))
Transport = m_true.Mtot
TransEmis = F
    
# ------------------------------------------------------
# Prior run
# ------------------------------------------------------
    
m_prior = TracerModel(Parameters_prior,method='Upwind',initialvalue=0)
m_prior.integrateModel()
Obs_prior=m_prior.results[:,stations]
Cost_prior = 1/sigmaxe*sum((array(Obs_prior)-array(Obs_true))**2)+1/sigmaxa*sum((array(E_prior)-array(E_prior))**2)
Derivative = Adjoined(E_prior,E_prior)

# ------------------------------------------------------
# Test derivative adjoined model for Qth element of emission vector E
# ------------------------------------------------------

alpha=0.000000001
Q=15
E_test=np.zeros(nx)
E0=1
for j in range(0,nx):
    x=np.int(j*Dx)
    xvec[j]=x
    if len(where(np.array(sources_guess)==x)[0])>0:
        E_test[j]=E0
E_test[Q]=E_test[Q]+alpha
Parameter_test = {'xmax':xmax,'dx':Dx,'tmax':tmax,'dt':Dt,'u0':u0,'k':k,'E':E_test}
m_test = TracerModel(Parameter_test,method='Upwind',initialvalue=0)
m_test.integrateModel()
Obs_test=m_test.results[:,stations]
Cost_test = 1/sigmaxe*sum((array(Obs_test)-array(Obs_true))**2)+1/sigmaxa*sum((array(E_test)-array(E_prior))**2)

print('Cost function',Cost_prior)
print('Calculated-derivative:',Derivative[Q])
print('Test-derivative:', (Cost_test-Cost_prior)/alpha)

# ------------------------------------------------------
# Implement DE (=Delta E)
# ------------------------------------------------------

DE=zeros(nx)-0.000000000000000001
E_new=np.zeros(nx)
E0=1
for j in range(0,nx):
    x=np.int(j*Dx)
    xvec[j]=x
    if len(where(np.array(sources_guess)==x)[0])>0:
        E_new[j]=E0
E_new=E_new+DE*Derivative
Parameter_new = {'xmax':xmax,'dx':Dx,'tmax':tmax,'dt':Dt,'u0':u0,'k':k,'E':E_new}
m_new = TracerModel(Parameter_new,method='Upwind',initialvalue=0)
m_new.integrateModel()
Obs_new=m_new.results[:,stations]
Cost_new = 1/sigmaxe*sum((array(Obs_new)-array(Obs_true))**2)+1/sigmaxa*sum((array(E_new)-array(E_prior))**2)
print('Cost function one time',Cost_new)

# ------------------------------------------------------
# Run Adjoined model
# ------------------------------------------------------

E_old=np.zeros(len(E_prior))
E_new=E_prior
K=1
i=0
j=0
meandif_vec=np.zeros(J)
absmeanadj_vec=np.zeros(J)


while j<J and K>mean(abs(E_new - E_old)) or i<=1:
    K=mean(abs(Adjoined(E_old,E_new)))
    E_old = E_new
    E_new = E_new - epsilon * Adjoined(E_old,E_new)
    meandif_vec[j]=mean(abs(E_new - E_old))
    absmeanadj_vec[j]=mean(abs(Adjoined(E_old,E_new)))
    j=j+1
    
    if np.mod(j,J/10.)==0:
        i=i+1
        print(10*i,'%')

E_final=E_new
Parameter_final = {'xmax':xmax,'dx':Dx,'tmax':tmax,'dt':Dt,'u0':u0,'k':k,'E':E_final}
m_final = TracerModel(Parameter_final,method='Upwind',initialvalue=0)
m_final.integrateModel()
Obs_final=m_new.results[:,stations]
Cost_final = 1/sigmaxe*sum((array(Obs_final)-array(Obs_true))**2)+1/sigmaxa*sum((array(E_final)-array(E_prior))**2)
print('Cost function difference',Cost_final-Cost_new)

# ------------------------------------------------------
# Plot
# ------------------------------------------------------

matplotlib.style.use('ggplot')
fig=plt.figure(num=None, figsize=(7,3),dpi=150, facecolor='w', edgecolor='k') # little: 5,3, large: 9,3
plt.scatter(stations,np.zeros(len(stations))+0.03,s=100,c='orange',alpha=1,zorder=15,edgecolor='k',linewidth=2,label='Measuring Stations')
plt.plot(E_final,label='Determined',linewidth=4)
plt.plot(E_prior,label='Prior',linewidth=4)
plt.plot(E_true,'dimgray',label='Actual',linewidth=2)
plt.xlabel('Distance x',fontsize=15)
plt.ylabel('Emission strength',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlim([0,nx])
plt.ylim([-0.3,1])
plt.legend(fontsize=9)
fig.tight_layout()
plt.show()

fig=plt.figure(num=None, figsize=(7,3),dpi=150, facecolor='w', edgecolor='k') # little: 5,3, large: 9,3
plt.semilogy(meandif_vec,linewidth=4)
plt.xlabel('Adjoined iteration step',fontsize=15)
plt.ylabel('Mean absolute difference',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
fig.tight_layout()
plt.show()

fig=plt.figure(num=None, figsize=(7,3),dpi=150, facecolor='w', edgecolor='k') # little: 5,3, large: 9,3
plt.semilogy(absmeanadj_vec,linewidth=4)
plt.xlabel('Adjoined iteration step',fontsize=15)
plt.ylabel('Mean absolute derivative',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
fig.tight_layout()
plt.show()