# ------------------------------------------------------
# Preambule
# ------------------------------------------------------

from numpy import *

# ------------------------------------------------------
# Define truth
# ------------------------------------------------------

em = array([1.,3.])
c = array([2.,3.])
print('truth:',c,em)

# ------------------------------------------------------
# Gain measurements (observations)
# ------------------------------------------------------

tp = array([[0,1],[1,0]])
c = array(c)
obst = []
for times in range(2):
    c = c + em
    c = matmul(tp,c)
    obst.extend([c[0],c[1]])
print('observations:', obst)

# ------------------------------------------------------
# First guess
# ------------------------------------------------------

em = array([2.,1.])
c = array([0.,0.])
print('start value:',c,em)

# ------------------------------------------------------
# Simulation results with first guess
# ------------------------------------------------------

obs = []
for times in range(2):
    c = c + em
    c = matmul(tp,c)
    obs.extend([c[0],c[1]])
print('simulation obs:', obs)
cost = sum((array(obs)-array(obst))**2)
print('cost function:',cost)
forcing = array(obs)-array(obst)  # (Hx-y)
print('forcing Hx-y:',forcing)

# ------------------------------------------------------
# Adjoined model
# ------------------------------------------------------

adc = array([0.,0.])
ade = array([0.,0.])
for times in [1,0]:
    adc = adc + forcing[times*2:times*2+2]
    adc = matmul(tp,adc)
    ade = ade + adc
print('derivative?:',2*adc,2*ade)

# ------------------------------------------------------
# Test derivative for c[0]
# ------------------------------------------------------

for alpha in [0.1,0.01,0.001,0.0001]:
    em = array([2.,1.])
    c = array([0.0+alpha,0.0])
    obs = []
    for times in range(2):
        c = c + em
        c = matmul(tp,c)
        obs.extend([c[0],c[1]])
    print('simulation obs:', obs) 
    costn = sum((array(obs)-array(obst))**2)
    print('cost function:',costn)
    print('derivative:', (costn-cost)/alpha)

# ------------------------------------------------------
# Test derivative for e[1]
# ------------------------------------------------------

for alpha in [0.1,0.01,0.001,0.0001]:
    em = array([2.,1.+alpha])
    c = array([0.0,0.0])
    obs = []
    for times in range(2):
        c = c + em
        c = matmul(tp,c)
        obs.extend([c[0],c[1]])
    print('simulation obs:', obs) 
    costn = sum((array(obs)-array(obst))**2)
    print('cost function:',costn)
    print('derivative:', (costn-cost)/alpha)