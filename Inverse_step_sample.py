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

indices = [10,30,50,80]
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
        K[ni] = matrix_power(F,ni)[i]

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
sigmaxa = 0.01      # prior
sigmaxe = 2e-8      # observations
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
# Plotting to compare
# ------------------------------------------------------

fig,ax = plt.subplots(1)
#fig,ax = newfig(0.8)
ax.plot(m1.x,x0,label='actual')
ax.plot(m1.x,x,label='determined')
ax.plot(indices,np.zeros(len(indices)),'ro',markersize=10)
ax.set_ylim(0,0.5)
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('E')
fig.tight_layout()
fig.show()
#savefig(fig,'simulation_plot_inverse')
#fig.savefig('simulation_plot_inverse2.pdf')