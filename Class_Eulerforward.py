#Preambule
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stat
import matplotlib
from numpy.linalg import inv
matplotlib.style.use('ggplot')
import csv
import matplotlib.cm as cm

class Eulerforward(object):

    def __init__(self,parameters,method='euler',initialvalue=None):
        self.P = dict.fromkeys(['xmax','Dx','tmax','Dt','u0','k','E','C'])
        self.P.update(parameters)
  
        # Set initial value to zero, with
        self.initialvalue = 1
        if initialvalue is not None:
            self.initialvalue = initialvalue

        if None in self.P.values():
            print('WARNING: SOME NECESSARY PARAMETERS HAVE NOT BEEN SPECIFIED. FIX BEFORE RUN')
            print('These parameters are undefined:')
            for key in self.P:
                if self.P[key] == None:
                    print(key)

    def updateParameters(self,newParams):
        '''
        Updates the model by loading new values into the parameter dictionary
        NOTE: this does not check if the parameters in newParams are the ones used by the model and/or in the correct data format
        '''
        self.P.update(newParams)

    def reportParameters(self):
        print('Reporting Model parameters')
        paramList = list(self.P.keys())
        paramList.sort()
        for key in paramList:
            print('{:.12s} , {:}'.format(key,self.P[key]))

    def initialize(self,initialvalue):
        '''
        Reinitialize the model with a new initial condition
        '''
        self.initialvalue = C
        
        
    def reportInitialState(self):   
        print('The initial state of the model is: later Ill do this')

    def integrateModel(self):
        '''
        Performs the simulation run
        '''
        import time as T
        start_time = T.time()
        print('================================================')
        print('Starting model run')
        print(' ')
        if None in self.P.values():
            sys.exit('SOME NECESSARY PARAMETERS HAVE NOT BEEN SPECIFIED. Aborting...')
        print(' ')
        self.reportInitialState()
        tlen = int(self.P['tmax'] / self.P['Dt'])
        xlen = int(self.P['tmax'] / self.P['Dx'])
		
        # Initialize time array
        self.time = np.linspace(0,self.P['tmax'],tlen)

        C_matrix = np.zeros(shape=(tlen,xlen)) # glacier length
		
        for j in range(0,xlen):
            C_matrix[0,j]=self.P['C'][j]
            
        for t in range(1,tlen):
            for j in range(0,xlen):
                C_matrix[t,j]=C_matrix[t-1,j]-self.P['u0'][j]*(C_matrix[t-1,j]-C_matrix[t-1,j-1])*self.P['Dt']/self.P['Dx']+self.P['E'][j]*self.P['Dt']-self.P['k']*C_matrix[t-1,j]*self.P['Dt']


        self.results = C_matrix

        print('Total time required: %.2f seconds' % (T.time() - start_time))
        print('')
        print('Model run finished')
        print('================================================')