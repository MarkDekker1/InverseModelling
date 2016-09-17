#Preambule
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stat
import matplotlib
from numpy.linalg import inv
matplotlib.style.use('ggplot')
import csv
import matplotlib.cm as cm

#Constants
xmax=100
Dx=1
xlen=np.int(xmax/Dx)
tmax=100
Dt=0.01
tlen=np.int(tmax/Dt)
xvec=np.zeros(100)
u0=10
k=0.1

#Set up E
E=np.zeros(100)
E0=1
for j in range(0,xlen):
    x=j*Dx
    xvec[j]=x
    if x==25 or x==50:
        E[j]=E0
    
#Set up C
C=np.zeros(100)

#%%
import time
start_time = time.time()

C_matrix=np.zeros(shape=(tlen,xlen))

for j in range(0,xlen):
    C_matrix[0,j]=C[j]
    
for t in range(1,tlen):
    for j in range(0,xlen):
        C_matrix[t,j]=C_matrix[t-1,j]-u0*(C_matrix[t-1,j]-C_matrix[t-1,j-1])*Dt/Dx+E[j]*Dt-k*C_matrix[t-1,j]*Dt
            
print("--- %s seconds ---" % (time.time() - start_time))

plt.figure(num=None, figsize=(10,4),dpi=150, facecolor='w', edgecolor='k')
plt.plot(xvec,C_matrix[np.int(tmax/Dt)-1], 'k-',linewidth=2)
plt.ylabel(r'Concentration',fontsize=15)
plt.xlabel('Distance [m]',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)

#%% Animation
from matplotlib import animation

fig = plt.figure()
ax = plt.axes(xlim=(0, xmax), ylim=(0,0.5))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = xvec
    y = C_matrix[i]
    line.set_data(x, y)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=25000, interval=0.01, blit=True)

plt.show()