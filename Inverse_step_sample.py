# ------------------------------------------------------
# Inverse Modelling using Tracer Class (variable station amount)
# ------------------------------------------------------

from Class_Tracer import *
from numpy.linalg import matrix_power,inv

# ------------------------------------------------------
# Constants
# ------------------------------------------------------

xmax    =   100
Dx      =   1
nx      =   np.int(xmax/Dx)
tmax    =   100
Dt      =   0.1
nt      =   np.int(tmax/Dt)
xvec    =   np.zeros(100)
k       =   0.1
u0      =   5
E0      =   1 # source strength

# ------------------------------------------------------
# Emissions
# ------------------------------------------------------

E   =   np.zeros(100)
for j in range(0,nx):
    x=j*Dx
    xvec[j]=x
    if x==1 or x==10 or x==50:
        E[j]=E0
        
# ------------------------------------------------------
# Forward model (Tracer class)
# ------------------------------------------------------
        
Parameter_initial = {
    'xmax':xmax,
    'dx':Dx,
    'tmax':tmax,
    'dt':Dt,
    'u0':u0,
    'k':k,
    'E':E
    }

m1 = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
m1.integrateModel()

# ------------------------------------------------------
# Choose which observation stations (their indices)
# ------------------------------------------------------

#indices = [10,30,50,80] # Example 1
#indices = [20,40,60,80] # Example 2
indices = [30,40,80,95] # Example 3
numindices = len(indices)
Khatlarge = np.zeros((numindices*nt,nx))
yinlarge = np.zeros(numindices*nt)

# ------------------------------------------------------
# Construct matrix F (relates x_0 = [c_i,E_i] to x_n = [c_n,E_i])
# ------------------------------------------------------

M = np.zeros((200,200))                       
M[:100,:100] = m1.Mtot
M[:100,100:] = np.diag(np.ones(100)*m1.P['dt'])
M[100:,100:] = np.diag(np.ones(100))

# ------------------------------------------------------
# Gain initial state vector
# ------------------------------------------------------

x0 = m1.P['E']

# ------------------------------------------------------
# Construct matrix K (relates x_0 to time series yin)
# ------------------------------------------------------

for numid in range(numindices):
    print('Processing timeseries %i ' % numid)
    i = indices[numid]
    yinlarge[numid*nt:(numid+1)*nt] = m1.results[:,i]
    
    K = np.zeros((nt,2*nx))
    for ni in range(nt):
        K[ni] = matrix_power(M,ni)[i]

    Khat = K[:,100:]
    
    Khatlarge[numid*nt:(numid+1)*nt,:] = Khat
   
# ------------------------------------------------------
# Test matrix K by calculating time series yt
# ------------------------------------------------------
   
ytlarge = np.matmul(Khatlarge,x0)
print('Max deviation y_t = %.3e' % abs(ytlarge-yinlarge).max())
    

# ------------------------------------------------------
# Create prior guess and error estimates
# ------------------------------------------------------

xa = np.zeros(100)  # all zeros, no prior knowledge
#xa = m1.P['E']     # assign sources, then guess is perfect
sigmaxa = 0.001      # prior (0.001)
sigmaxe = 2e-8      # observations (2e-8)
Sa = np.diag(sigmaxa*np.ones(nx))
Se = np.diag(sigmaxe*np.ones(nt*numindices))
print('Computing best estimate')

# ------------------------------------------------------
# Construct matrix G (Jacob, Eqn. 5.9)
# ------------------------------------------------------

G = np.matmul(np.matmul(inv(np.matmul(np.matmul(Khatlarge.transpose(),inv(Se)),Khatlarge)+inv(Sa)),Khatlarge.transpose()),inv(Se))

# ------------------------------------------------------
# Initial vector based on yin, K and prior (Jacob, Enq. 5.7)
# ------------------------------------------------------

x = xa + np.matmul(G,yinlarge-np.matmul(Khatlarge,xa))

# ------------------------------------------------------
# Error covariance matrix (Jacob, Enq. 5.10)
# ------------------------------------------------------

Shat = inv(np.matmul(np.matmul(Khatlarge.transpose(),inv(Se)),Khatlarge) + inv(Sa))

# ------------------------------------------------------
# Inverse modelling correctness score
# ------------------------------------------------------

RMSE=np.mean(np.sqrt((x-x0)**2))
print(RMSE)

# ------------------------------------------------------
# Plotting to compare
# ------------------------------------------------------

matplotlib.style.use('ggplot')
fig=plt.figure(num=None, figsize=(5,3),dpi=150, facecolor='w', edgecolor='k') # little: 5,3, large: 9,3
plt.scatter(indices,np.zeros(len(indices))+0.03,s=100,c='orange',alpha=1,zorder=15,edgecolor='k',linewidth=2,label='Measuring Stations')
plt.plot(m1.x,x,'mediumvioletred',label='Determined',linewidth=4)
plt.plot(m1.x,x0,'dimgray',label='Actual',linewidth=2)
plt.xlabel('Distance x',fontsize=15)
plt.ylabel('Emission strength',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlim([0,100])
plt.ylim([0,1])
plt.legend(loc='best',fontsize=9)
fig.tight_layout()
plt.show()
#savefig(fig,'simulation_plot_inverse')
#fig.savefig('simulation_plot_inverse2.pdf')