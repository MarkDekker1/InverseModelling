# --------------------------------------------------
# Constants
# --------------------------------------------------

import numdifftools as nd
xmax=100
Dx=1
xlen=np.int(xmax/Dx)
tmax=100
Dt=0.1
tlen=np.int(tmax/Dt)
xvec=np.zeros(100)
k=0.1
u0 = 5

E=np.zeros(100)
C0=np.zeros(100)
E0=1
for j in range(0,xlen):
    x=j*Dx
    xvec[j]=x
    if x==1 or x==10 or x==50:
        E[j]=E0

E_p=np.zeros(100)
E0=1
var_point=3
x_first=np.int(1+np.random.normal(0,var_point))
x_second=np.int(10+np.random.normal(0,var_point))
x_third=np.int(50+np.random.normal(0,var_point))
for j in range(0,xlen):
    x=j*Dx
    xvec[j]=x
    if x==x_first or x==x_second or x==x_third:
        E_p[j]=E0
    E_p[j]=E_p[j]+np.random.normal(0,E0/100.)

# --------------------------------------------------
# Prior run
# --------------------------------------------------

Parameter_initial = {'xmax':xmax,
    'dx':Dx,
    'tmax':tmax,
    'dt':Dt,
    'u0':u0,
    'k':k,
    'E':E_p}

M_p = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
M_p.integrateModel()

# --------------------------------------------------
# Actual
# --------------------------------------------------

Parameter_initial = {'xmax':xmax,
    'dx':Dx,
    'tmax':tmax,
    'dt':Dt,
    'u0':u0,
    'k':k,
    'E':E}

M_i = TracerModel(Parameter_initial,method='Upwind',initialvalue=0)
M_i.integrateModel()

# --------------------------------------------------
# Uncertainties
# --------------------------------------------------

S_p=np.zeros(shape=(100,100))
for i in range(0,100):
    for j in range(0,100):
        if i==j:
            S_p[i,j]=0.25
            
S_i=np.zeros(shape=(1000,1000))
for i in range(0,1000):
    for j in range(0,1000):
        if i==j:
            S_i[i,j]=0.1

# --------------------------------------------------
# Result at different stations (including plot)
# --------------------------------------------------

x_1=1
x_2=30
x_3=60
x_4=90

yp_1=M_p.results[:,x_1-1]
yp_2=M_p.results[:,x_2-1]
yp_3=M_p.results[:,x_3-1]
yp_4=M_p.results[:,x_4-1]

yi_1=M_i.results[:,x_1-1]
yi_2=M_i.results[:,x_2-1]
yi_3=M_i.results[:,x_3-1]
yi_4=M_i.results[:,x_4-1]

plt.figure(figsize=(8,6))
plt.plot(yp_1,'r')
plt.plot(yi_1,'r--')
plt.plot(yp_2,'b')
plt.plot(yi_2,'b--')
plt.plot(yp_3,'k')
plt.plot(yi_3,'k--')
plt.plot(yp_4,'g')
plt.plot(yi_4,'g--')

plt.xlabel('time')
plt.ylabel('concentration')
plt.show()

# --------------------------------------------------
# Define and calculate J
# --------------------------------------------------

def CostJ(E0,E_p0,S_p0,S_i0,yp_0,yi_0):
    return np.dot(np.dot(np.transpose(E0-E_p0),np.linalg.inv(S_p0)),(E0-E_p0))+np.dot(np.dot(np.transpose(yp_0-yi_0),np.linalg.inv(S_i0)),(yp_0-yi_0))

J1=CostJ(E,E_p,S_p,S_i,yp_1,yi_1)
J2=CostJ(E,E_p,S_p,S_i,yp_2,yi_2)
J3=CostJ(E,E_p,S_p,S_i,yp_3,yi_3)
J4=CostJ(E,E_p,S_p,S_i,yp_4,yi_4)

TotalJ=J1+J2+J3+J4
print TotalJ
#%%
def PDF(S_p0,E0,E_p0):
    return np.exp(-0.5*np.dot(np.dot(np.transpose(E0-E_p0),np.linalg.inv(S_p0)),(E0-E_p0)))
    #1/((2.*np.pi)**(len(yi_0)/2)*np.linalg.det(S_p0)**(0.5))*


#%%
A=np.zeros(shape=(100,100))
for i in range(0,100):
    for j in range(0,100):
        if i==j:
            A[i,j]=1+(-u0/Dx-k)*Dt
        if i!=0:
            if j==i-1:
                A[i,j]=u0*Dt/Dx
        if i==0:
            if j==99:
                A[i,j]=u0*Dt/Dx

B=np.identity(100)
for i in range(1,1000-1):
    B=B+A**i

C=A**1000

K=np.zeros(shape=(200,200))
for i in range(0,200):
    if i<100:
        for j in range(0,100):
            K[i,j]=B[i,j]
    else:
        for j in range(100,200):
            K[i,j]=C[i-100,j-100]

x=np.zeros(200)
for i in range(0,200):
    if i<100:
        x[i]=E[i]
    else:
        x[i]=C0[i-100]

#%%
y=np.dot(K,x)