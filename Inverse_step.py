
# Inverse Modelling using Tracer Class

#%matplotlib
from Class_Tracer import *

from numpy.linalg import matrix_power,inv
import scipy.linalg as la


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
F = np.zeros((200,200))                       
F[:100,:100] = m1.Mtot                         
F[:100,100:] = np.diag(np.ones(100)*m1.P['dt'])
F[100:,100:] = np.diag(np.ones(100))           

# construct matrix relating initial vector x_0 to timeseries yin
# x_{i,ti} = K_i x^0
K = np.zeros((nt,2*nx))
for ni in range(nt):
    K[ni] = matrix_power(F,ni)[i]

# Take matrix Khat that takes only E as input
# x_{i,ti} = Khat_i E
Khat = K[:,100:]

# Actual initial vector
#x0 = np.zeros(2*nx)
#x0[100:] = m1.P['E']

x0 = m1.P['E']

# Test if matrix F works
# compute final profile x_N
xnt = np.matmul(matrix_power(F,m1.P['nt']),x0)
print('Max deviation x_N = %.3e' % abs(xnt[:nx]-m1.results[-1,:]).max())

# Test if matrix K works
# timeseries computed using K, x0
yt = np.matmul(K,x0)
print('Max deviation y_t = %.3e' % abs(yt-yin).max())

# Test if matrix Khat works
# timeseries computed using Khat, E
yt2 = np.matmul(Khat,m1.P['E'])
print('Max deviation y_t = %.3e' % abs(yt-yin).max())

# Prior guess for vector x_0
# HERE FOR USE WITH K
#xa = np.zeros(200) # all zeros, no prior knowledge
# if desired, provide perfect prior guess
#xa[100:] = m1.P['E'] # assign sources, then guess is perfect

# HERE FOR USE WITH Khat
xa = np.zeros(100) # all zeros, no prior knowledge
#xa = m1.P['E'] # assign sources, then guess is perfect

# error estimates 
sigmaxa = 0.001#10        #e-9 #20#0.00001 # error in the prior
sigmaxe = 2e-8      #0.00000000001 # error in the observations

#Sa = np.diag(sigmaxa*np.ones(2*nx))
Sa = np.diag(sigmaxa*np.ones(nx))
Se = np.diag(sigmaxe*np.ones(nt))

# construct G matrix (Jacob, eq. 5.9)
#G = np.matmul(np.matmul(inv(np.matmul(np.matmul(K.transpose(),inv(Se)),K)+inv(Sa)),K.transpose()),inv(Se))
G = np.matmul(np.matmul(inv(np.matmul(np.matmul(Khat.transpose(),inv(Se)),Khat)+inv(Sa)),Khat.transpose()),inv(Se))
# compute initial vector based on yin, K, xa (Jacob, eq. 5.7)
x = xa + np.matmul(G,yin-np.matmul(Khat,xa))

# Error covariance of initial vector (Jacob, eq. 5.10)
Shat = inv(np.matmul(np.matmul(Khat.transpose(),inv(Se)),Khat) + inv(Sa))

# Compare with actual initial vector
fig,ax = plt.subplots(1)
ax.plot(x0,label='actual')
ax.plot(x,label='determined')
ax.set_ylim(0,0.5)
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('E')


# =====================================================
# Inverse step using multiple timeseries

# Construct Matrices 
indices = [10,30,50,80]
numindices = len(indices)

Khatlarge = np.zeros((numindices*nt,nx))
yinlarge = np.zeros(numindices*nt)

x0 = m1.P['E'] # source vector

for numid in range(numindices):
    print(numid)
    i = indices[numid]
    yinlarge[numid*nt:(numid+1)*nt] = m1.results[:,i]
    
    K = np.zeros((nt,2*nx))
    for ni in range(nt):
        K[ni] = matrix_power(F,ni)[i]

    Khat = K[:,100:]
    
    Khatlarge[numid*nt:(numid+1)*nt,:] = Khat
   
# Test if matrix Khatlarge works
# timeseries computed using Khat, E
ytlarge = np.matmul(Khatlarge,x0)
print('Max deviation y_t = %.3e' % abs(ytlarge-yinlarge).max())
    
# Compute best estimate

xa = np.zeros(100) # all zeros, no prior knowledge
#xa = m1.P['E'] # assign sources, then guess is perfect

# error estimates 
sigmaxa = 0.01  # error in the prior
sigmaxe = 2e-8  # error in the observations

#Sa = np.diag(sigmaxa*np.ones(2*nx))
Sa = np.diag(sigmaxa*np.ones(nx))
Se = np.diag(sigmaxe*np.ones(nt*numindices))

# construct G matrix (Jacob, eq. 5.9)
G = np.matmul(np.matmul(inv(np.matmul(np.matmul(Khatlarge.transpose(),inv(Se)),Khatlarge)+inv(Sa)),Khatlarge.transpose()),inv(Se))
# compute initial vector based on yin, K, xa (Jacob, eq. 5.7)
x = xa + np.matmul(G,yinlarge-np.matmul(Khatlarge,xa))

# Error covariance of initial vector (Jacob, eq. 5.10)
Shat = inv(np.matmul(np.matmul(Khatlarge.transpose(),inv(Se)),Khatlarge) + inv(Sa))

# Compare with actual initial vector
fig,ax = plt.subplots(1)
#fig,ax = newfig(0.8)
ax.plot(m1.x,x0,label='actual')
ax.plot(m1.x,x,label='determined')
ax.plot([10,30,50,80],np.zeros(4),'ro',markersize=10)
ax.set_ylim(0,0.5)
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('E')
fig.tight_layout()
#savefig(fig,'simulation_plot_inverse')
#fig.savefig('simulation_plot_inverse2.pdf')