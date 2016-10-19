# CHEMICAL TRACER MODELLING
# BY: Mark Dekker, Matthias Aengenheyster

# CONTENTS:
#-Class TracerModel

#Preambule
import sys,os
import time as T

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import csv
import matplotlib.cm as cm

class TracerModel(object):

    def __init__(self,parameters,method='Upwind',initialvalue=None):
        '''
        Initialization method. 

        INPUTS:
            -parameters: dictionary with model parameters: Requires. ['xmax','dx','tmax','dt','u0','k','E']
                xmax    : length of x-dimension, float
                dx      : spatial resolution, float
                tmax    : length of integration, float
                dt      : timestep, float
                u0      : windspeed, float
                k       : decay constant. float or array of the correct size (nx = xmax / dx)
                E       : sources. array of the correct size (nx = xmax / dx)
            -method: integration rule. Supports 'Upwind','Leapfrog', 'LaxWendroff'. by default 'Upwind'
            -initialvalue: initial condition. array of the correct size (nx = xmax / dx)
        OUTPUT: no output variables

        Does the following:
            -Assign parameters from the dictionary parameters to self.P
            -Set self.initialvalue to its value, or to 0, if not specified
            -Set integration method (by default Upwind)
            -Check if all essential parameters have been passed from parameters
            -call the method self.initialize
                -computes derived parameters
                -initializes arrays for time, space and results
                -assigns initial condition to first entry of results
                -checks whether initialvalue is either a single float or an array of size nx. If not, breaks.
        '''
        self.P = dict.fromkeys(['xmax','dx','tmax','dt','u0','k','E'])
        self.P.update(parameters)
  
        # Set initial value to zero, if not specified
        self.initialvalue = 0
        if initialvalue is not None:
            self.initialvalue = initialvalue

        # Set integration rule to Upwind + Euler, if not otherwise specified
        self.method = method

        if None in self.P.values():
            print('WARNING: SOME NECESSARY PARAMETERS HAVE NOT BEEN SPECIFIED. FIX BEFORE RUN')
            print('These parameters are undefined:')
            for key in self.P:
                if self.P[key] == None:
                    print(key)

        self.initialize()
        #print('Model initialized')

    def initialize(self):
        '''
        Initialization helper routine

        INPUTS: none (except for self)
        OUTPUTS: none

        computes derived parameters and creates arrays for time, space, and results
        assigns initial condition to first entry of results array
            -computes derived parameters
            -initializes arrays for time, space and results
            -assigns initial condition to first entry of results
            -checks whether initialvalue is either a single float or an array of size nx. If not, breaks.
        '''

        # initialize arrays
        self.x = np.arange(0,self.P['xmax'],self.P['dx'])
        self.time = np.arange(0,self.P['tmax'],self.P['dt'])
        # compute derived parameters
        self.P['nx'] = self.x.shape[0]
        self.P['nt'] = self.time.shape[0]
        self.P['gamma'] = self.P['u0'] * self.P['dt'] / self.P['dx']
        # initialize results array
        self.results = np.zeros((self.P['nt'],self.P['nx']))

        # set initial condition, if correct dimension
        try:
            self.results[0,:] = self.initialvalue
        except ValueError:
            sys.exit('Initial value has wrong dimension: %s instead of %s' % (str(self.initialvalue.shape),str(self.results[0,:])))

        #initialvalue = np.array(self.initialvalue)
        #if not len(np.array(self.initialvalue)) in [1,self.P['nx']]:
        #    sys.exit('Initial value has wrong dimension: %s instead of %s' % (str(self.initialvalue.shape),str(self.results[0,:])))
        #else:
        #    self.results[0,:] = self.initialvalue
        #print('Done with initialize')


    def updateParameters(self,newParams):
        '''
        INPUTS: newParams, dictionary

        Updates the model by loading new values into the parameter dictionary
        NOTE: this does not check if the parameters in newParams are the ones used by the model and/or in the correct data format
        '''
        self.P.update(newParams)
        self.initialize() # recompute derived parameters

    def reportParameters(self):
        print('Reporting Model parameters')
        paramList = list(self.P.keys())
        paramList.sort()
        for key in paramList:
            print('{:.12s} , {:}'.format(key,self.P[key]))

    def setInitial(self,initialvalue):
        '''
        Reinitialize the model with a new initial condition
        then calls self.initialize
        '''
        self.initialvalue = initialvalue
        self.initialize()


    def reportInitialState(self):   
        print('The initial state of the model is: later Ill do this')


    def set_method(self,method):
        '''
        set integration rule.
        available:
            -Upwind : Upwind in space, Euler forward in time
            -Leapfrog : Leapfrog in space, Euler forward in time
            -LaxWendroff: Lax-Wendroff
        '''
        possibleMethods = ['Upwind','Leapfrog','LaxWendroff']
        if method not in possibleMethods:
            print('Integration method incorrectly specified. Method "%s" does not exist' % method)
            print('Possible methods are %s' % (str(possibleMethods)))
            sys.exit('Exiting ...')
        self.method = method

    def get_matrix(self,bm1,b0,bp1,k,dt,nx):
        '''
        Compute forward matrix (Advection + decay) based on diagonal elements
        c^{n+1} = Mtot c^{n}
        for Leapfrog additionally need matrix for step n-1 (not done here)
        '''
        Madv = np.diag(bm1 * np.ones(nx-1),k=-1) + np.diag(b0 * np.ones(nx),k=0) + np.diag(bp1 * np.ones(nx-1),k=1)
        Madv[0,-1] = bm1
        Madv[-1,0] = bp1
        # Decay matrix
        kj = k * dt * np.ones(nx)
        Mdec = np.diag(kj,k=0)
        # combine matrices
        Mtot = Madv - Mdec
        return Mtot


    def integrateModel(self):
        '''
        Integrate the model based on
            -specified parameters, including source array
            -specified initial condition
            -specified integration method
        Results are saved in self.results
        '''

        # test if E has correct dimension
        if not self.P['E'].shape == self.results[0,:].shape:
            sys.exit('Source array does not have correct shape: %s . Needs same shape as x-dimemsion: %s' % (str(self.P['E']), str(self.results[0,:].shape)))

        # =====================================
        # METHOD SELECTION LOOP
        # =====================================
        if self.method == 'Upwind':
            bm1 = 0.5 * self.P['gamma']
            b0 = 1 - 0.5 * self.P['gamma']
            bp1 = 0
            Mtot = self.get_matrix(bm1,b0,bp1,self.P['k'],self.P['dt'],self.P['nx'])

            # assign matrices to properties
            self.Mtot = Mtot
            self.M2 = 0

            # MODEL RUN
            start_time = T.time()
            #print('================================================')
            #print('Starting model run with method %s' % self.method)

            for ti in range(0,self.P['nt']-1):
                self.results[ti+1,:] = np.matmul(Mtot,self.results[ti,:]) + self.P['dt'] * self.P['E']

                #if np.mod(ti,np.int(self.P['nt']/10))==0:
                    #print('Progress is at ', ti/self.P['nt']*100., 'percent')

            #print('Total time required: %.2f seconds' % (T.time() - start_time))
            #print('Model run finished')
            #print('================================================')
            # END OF MODEL RUN

        elif self.method == 'LaxWendroff':
            bm1 = self.P['gamma'] / 2 * (self.P['gamma'] + 1)
            b0 = 1 - self.P['gamma']**2
            bp1 = self.P['gamma'] / 2 * (self.P['gamma'] -1)
            Mtot = self.get_matrix(bm1,b0,bp1,self.P['k'],self.P['dt'],self.P['nx'])

            # assign matrices to properties
            self.Mtot = Mtot
            self.M2 = 0

            # MODEL RUN
            start_time = T.time()
            print('================================================')
            print('Starting model run with method %s' % self.method)

            for ti in range(0,self.P['nt']-1):
                self.results[ti+1,:] = np.matmul(Mtot,self.results[ti,:]) + self.P['dt'] * self.P['E']

                if np.mod(ti,np.int(self.P['nt']/10))==0:
                    print('Progress is at ', ti/self.P['nt']*100., 'percent')
            print('Total time required: %.2f seconds' % (T.time() - start_time))
            print('Model run finished')
            print('================================================')
            # END OF MODEL RUN

        elif self.method == 'Leapfrog':
            bm1 = self.P['gamma']
            b0 = 0
            bp1 = - self.P['gamma']
            Mtot = self.get_matrix(bm1,b0,bp1,self.P['k'],2*self.P['dt'],self.P['nx'])
            M2 = np.diag(np.ones(self.P['nx'])) # matrix for step n-1

            # assign matrices to properties
            self.Mtot = Mtot
            self.M2 = M2

            # MODEL RUN
            start_time = T.time()
            print('================================================')
            print('Starting model run with method %s' % self.method)

            # For Leapfrog, start with one initial Euler step
            MEuler = self.get_matrix(0.5*self.P['gamma'],1-0.5*self.P['gamma'],0,self.P['k'],self.P['dt'],self.P['nx'])
            self.MEuler = MEuler
            self.results[1,:] = np.matmul(MEuler,self.results[0,:]) + self.P['dt'] * self.P['E']
            # all other steps using Leapfrog
            for ti in range(1,self.P['nt']-1):
                self.results[ti+1,:] = np.matmul(Mtot,self.results[ti,:]) + np.matmul(M2,self.results[ti-1,:]) + 2*self.P['dt'] * self.P['E']

                if np.mod(ti,np.int(self.P['nt']/10))==0:
                    print('Progress is at ', ti/self.P['nt']*100., 'percent')
            print('Total time required: %.2f seconds' % (T.time() - start_time))
            print('Model run finished')
            print('================================================')
            # END OF MODEL RUN

        else:
            sys.exit('Integration method incorrectly specified. Method "%s" does not exist' % self.method)

        # =====================================


    def saveModel(self,savename):
        '''
        INPUTS: savename: data is saved in the file 'savename.npy'

        saves model
            -parameters self.P
            -method, initialvalue
            -x,time, results
        '''
        container = self.P.copy()

        container['method'] = self.method
        container['initialvalue'] = self.initialvalue
        container['Mtot'] = self.Mtot
        container['M2'] = self.M2
        container['x'] = self.x
        container['time'] = self.time
        container['results'] = self.results

        np.save(savename + '.npy', container)
        print('Saving done')