# ------------------------------------------------------
# Inverse Modelling using Tracer Class
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
# Choose which observation station (its index)
# ------------------------------------------------------

i = 5#30#70
yin = m1.results[:,i]

# ------------------------------------------------------
# Construct matrix F (relates x_0 = [c_i,E_i] to x_n = [c_n,E_i])
# ------------------------------------------------------

M = np.zeros((200,200))                       
M[:100,:100] = m1.Mtot
M[:100,100:] = np.diag(np.ones(100)*m1.P['dt'])
M[100:,100:] = np.diag(np.ones(100))

# ------------------------------------------------------
# Construct matrix K (relates x_0 to time series yin)
# ------------------------------------------------------

K = np.zeros((nt,2*nx))
for ni in range(nt):
    K[ni] = matrix_power(M,ni)[i]

# ------------------------------------------------------
# Gain initial state vector (length 2*nx)
# ------------------------------------------------------

x0 = np.zeros(2*nx)
x0[100:] = m1.P['E']

# ------------------------------------------------------
# Test matrix F by calculating x^n
# ------------------------------------------------------

xnt = np.matmul(matrix_power(M,m1.P['nt']),x0)
print('Max deviation x_N = %.3e' % abs(xnt[:nx]-m1.results[-1,:]).max())

# ------------------------------------------------------
# Test matrix K by calculating time series yt
# ------------------------------------------------------

yt = np.matmul(K,x0)
print('Max deviation y_t = %.3e' % abs(yt-yin).max())

# ------------------------------------------------------
# Create prior guess and error estimates
# ------------------------------------------------------

xa = np.zeros(200)
#xa[100:] = m1.P['E']

sigmaxa = 0.001     # in prior
sigmaxe = 2e-8      # in observations

Sa = np.diag(sigmaxa*np.ones(2*nx))
Se = np.diag(sigmaxe*np.ones(nt))

# ------------------------------------------------------
# Construct matrix G (Jacob, Eqn. 5.9)
# ------------------------------------------------------

G = np.matmul(np.matmul(inv(np.matmul(np.matmul(K.transpose(),inv(Se)),K)+inv(Sa)),K.transpose()),inv(Se))

# ------------------------------------------------------
# Initial vector based on yin, K and prior (Jacob, Enq. 5.7)
# ------------------------------------------------------

x = xa + np.matmul(G,yin-np.matmul(K,xa))

# ------------------------------------------------------
# Error covariance matrix (Jacob, Enq. 5.10)
# ------------------------------------------------------

Shat = inv(np.matmul(np.matmul(K.transpose(),inv(Se)),K) + inv(Sa))

# ------------------------------------------------------
# Plotting to compare
# ------------------------------------------------------

fig,ax = plt.subplots(1)
ax.plot(x0,label='actual')
ax.plot(x,label='determined')
ax.set_ylim(0,0.5)
ax.legend()
ax.set_xlabel('left half: c0, right half: E')
