
# Inverse Modelling using Tracer Class

#%matplotlib
from Class_Tracer import *

from numpy.linalg import matrix_power,inv

#Constants
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

#Set up source array E
E   =   np.zeros(100)
for j in range(0,nx):
    x=j*Dx
    xvec[j]=x
    #if x==25 or x==50 or x==75:
    if x==1 or x==10 or x==50:
        E[j]=E0

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

m1 = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
m1.integrateModel()

# Inverse Step

# observation station, index
i = 5#30#70

# station timeseries
yin = m1.results[:,i]

# construct large matrix, relates x_0 = [c_i,E_i] = x_n = [c_n,E_i]
M = np.zeros((200,200))                       
M[:100,:100] = m1.Mtot                         
M[:100,100:] = np.diag(np.ones(100)*m1.P['dt'])
M[100:,100:] = np.diag(np.ones(100))           

# construct matrix relating initial vector x_0 to timeseries yin
K = np.zeros((nt,2*nx))
for ni in range(nt):
    K[ni] = matrix_power(M,ni)[i]

# Actual initial vector
x0 = np.zeros(2*nx)
x0[100:] = m1.P['E']

# Test if matrix M works
# compute final profile x_N
xnt = np.matmul(matrix_power(M,m1.P['nt']),x0)
print('Max deviation x_N = %.3e' % abs(xnt[:nx]-m1.results[-1,:]).max())

# Test if matrix K works
# timeseries computed using K, x0
yt = np.matmul(K,x0)
print('Max deviation y_t = %.3e' % abs(yt-yin).max())

# Prior guess for vector x_0
xa = np.zeros(200) # all zeros, no prior knowledge
# if desired, provide perfect prior guess
#xa[100:] = m1.P['E'] # assign sources, then guess is perfect

# error estimates 
sigmaxa = 0.001#10        #e-9 #20#0.00001 # error in the prior
sigmaxe = 2e-8      #0.00000000001 # error in the observations

Sa = np.diag(sigmaxa*np.ones(2*nx))
Se = np.diag(sigmaxe*np.ones(nt))

# construct G matrix (Jacob, eq. 5.9)
G = np.matmul(np.matmul(inv(np.matmul(np.matmul(K.transpose(),inv(Se)),K)+inv(Sa)),K.transpose()),inv(Se))
# compute initial vector based on yin, K, xa (Jacob, eq. 5.7)
x = xa + np.matmul(G,yin-np.matmul(K,xa))

# Error covariance of initial vector (Jacob, eq. 5.10)
Shat = inv(np.matmul(np.matmul(K.transpose(),inv(Se)),K) + inv(Sa))

# Compare with actual initial vector
fig,ax = plt.subplots(1)
ax.plot(x0,label='actual')
ax.plot(x,label='determined')
ax.set_ylim(0,0.5)
ax.legend()
ax.set_xlabel('left half: c0, right half: E')
