
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