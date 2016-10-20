
# Sample Usage of Class_Tracer.py

#%matplotlib
from Class_Tracer import *

#Constants
xmax=100
Dx=1
xlen=np.int(xmax/Dx)
tmax=100
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
    #if x==25 or x==50 or x==75:
    if x==1 or x==10 or x==50:
        E[j]=E0

u0 = 5

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

#%% Animation
from matplotlib import animation

fig = plt.figure()
ax = plt.axes(xlim=(0, xmax), ylim=(0,1))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    time_text.set_text('time = 0.0')
    return line,time_text

def animate(i):
    x = xvec
    y = m1.results[i]
    line.set_data(x, y)
    time_text.set_text('time = %.1f' % i )
    return line,time_text

time_text = plt.text(0.1, 0.1, '', zorder=10)
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=25000, interval=20, blit=True)

plt.show()