#Preambule
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stat
import matplotlib
from numpy.linalg import inv
matplotlib.style.use('ggplot')
import csv
import matplotlib.cm as cm
import Class_Eulerforward as EF

#Constants
xmax=100
Dx=1
xlen=np.int(xmax/Dx)
tmax=1000
Dt=0.1
tlen=np.int(tmax/Dt)
xvec=np.zeros(100)
k=0.1

#Set up E
E=np.zeros(100)
E0=1
for j in range(0,xlen):
    x=j*Dx
    xvec[j]=x
    if x==25 or x==50 or x==75:
        E[j]=E0
        
#Set up u0
u0=np.zeros(100)
for j in range(0,xlen):
    u0[j]=np.abs(np.sin(j/xlen*2.*np.pi))*5.+5.
        
    
#Set up C
C=np.zeros(100)

Parameter_initial = {
    'xmax':xmax,
    'Dx':Dx,
    'tmax':tmax,
    'Dt':Dt,
    'u0':u0,
    'k':k,
    'E':E,
    'C':C
    }

Run = EF.Eulerforward(Parameter_initial,'euler')

Run.integrateModel()

#%%Plot

plt.figure(num=None, figsize=(10,4),dpi=150, facecolor='w', edgecolor='k')
plt.plot(xvec,Run.results[np.int(tmax/Dt)-1], 'k-',linewidth=2)
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
    time_text.set_text('time = 0.0')
    return line,time_text

def animate(i):
    x = xvec
    y = Run.results[i]
    line.set_data(x, y)
    time_text.set_text('time = %.1f' % i )
    return line,time_text

time_text = plt.text(0.1, 0.1, '', zorder=10)
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=25000, interval=20, blit=True)

plt.show()