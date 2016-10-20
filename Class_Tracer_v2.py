# CHEMICAL TRACER MODELLING
# BY: Mark Dekker, Matthias Aengenheyster

# CONTENTS:
#-Class TracerModel

#Preambule
import sys,os
import time as T

import numpy as np
from numpy.linalg import matrix_power,inv

import scipy.sparse as sp
import matplotlib.pyplot as plt

#import matplotlib
#matplotlib.style.use('ggplot')
#import matplotlib.cm as cm

# Graphics
def figsize(scale):
    fig_width_pt = 426.79135 #pt 278.83713 #469.755    # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

# I make my own newfig and savefig functions
def newfig(width,nr=1,nc=1):
    #plt.clf()
    fig,ax = plt.subplots(nrows=nr,ncols=nc,figsize=figsize(width),dpi=300)
    #fig = plt.figure(figsize=figsize(width))
    #fig.dpi = 300
    #ax = fig.add_subplot(111)
    return fig, ax

def savefig(fig,filename):
    fig.savefig('{}.pgf'.format(filename))
    fig.savefig('{}.pdf'.format(filename))

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
        print('Model initialized')

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
        self.P['nx'] = np.int(self.P['xmax']/self.P['dx'])
        self.P['nt'] = np.int(self.P['tmax']/self.P['dt'])
        self.x = self.P['dx'] * np.arange(self.P['nx'])
        self.time = self.P['dt'] * np.arange(self.P['nt'])

        #self.x = np.arange(0,self.P['xmax'],self.P['dx'])
        #self.time = np.arange(0,self.P['tmax'],self.P['dt'])
        # compute derived parameters
        #self.P['nx'] = self.x.shape[0]
        #self.P['nt'] = self.time.shape[0]
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
        print('Done with initialize')


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

    # ===================================
    # FORWARD PART

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
            #bm1 = 0.5 * self.P['gamma']
            #b0 = 1 - 0.5 * self.P['gamma']
            bm1 = self.P['gamma']
            b0 = 1 - self.P['gamma']
            bp1 = 0
            Mtot = self.get_matrix(bm1,b0,bp1,self.P['k'],self.P['dt'],self.P['nx'])

            # assign matrices to properties
            #self.Mtot = Mtot
            self.Mtot = sp.csc_matrix(Mtot)
            self.M2 = 0

            # MODEL RUN
            start_time = T.time()
            print('================================================')
            print('Starting model run with method %s' % self.method)

            for ti in range(0,self.P['nt']-1):
                #self.results[ti+1,:] = np.matmul(Mtot,self.results[ti,:]) + self.P['dt'] * self.P['E']
                self.results[ti+1,:] = Mtot.dot(self.results[ti,:].transpose()) + self.P['dt'] * self.P['E']

                if np.mod(ti,np.int(self.P['nt']/10))==0:
                    print('Progress is at ', ti/self.P['nt']*100., 'percent')

            print('Total time required: %.2f seconds' % (T.time() - start_time))
            print('Model run finished')
            print('================================================')
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

    # ===================================
    # INVERSE PART
    def inverseSetup(self,inverseParams):
        self.Pinv = dict.fromkeys(['stations','sigmaxa','sigmaxe','noiseadd','noisemult'])
        self.Pinv.update(inverseParams)
        if None in self.Pinv.values():
            sys.exit('Essential paramter undefined! Essential parameters are "stations","sigmaxa","sigmaxe","noiseadd","noisemult"')

    def inverseUpdate(self,updateParams):
        ''' Update inverse parameters'''
        self.Pinv.update(updateParams)

    def inverseKmatrix(self):
        '''compute large K matrix'''
        nx,nt = self.P['nx'],self.P['nt']
        indices = (np.array(self.Pinv['stations']) * nx//100).astype(int) # measurement stations
        F = np.zeros((2*nx,2*nx))
        F[:nx,:nx] = self.Mtot.toarray()
        F[:nx,nx:] = np.diag(np.ones(nx)*self.P['dt'])
        F[nx:,nx:] = np.diag(np.ones(nx))

        numindices = len(indices)

        Khatlarge = np.zeros((numindices*nt,nx))
        yinlarge = np.zeros(numindices*nt)

        for numid in range(numindices):
            print('Processing timeseries %i ' % (numid+1))
            i = indices[numid]
            yinlarge[numid*nt:(numid+1)*nt] = self.results[:,i]
            
            K = np.zeros((nt,2*nx))
            for ni in range(nt):
                K[ni] = matrix_power(F,ni)[i]
            Khat = K[:,nx:]
            Khatlarge[numid*nt:(numid+1)*nt,:] = Khat

        # add additive and multiplicative noise
        # uniform distribution
        #yinlarge = yinlarge * (1 + self.Pinv['noisemult'] * np.random.uniform(low=-1,high=1,size=len(yinlarge))) + self.Pinv['noiseadd'] * np.random.uniform(low=-1,high=1,size=len(yinlarge))
        # normal distribution
        if self.Pinv['noisemult'] == 0:
            self.Pinv['noisemult'] = 1e-20
        if self.Pinv['noiseadd'] == 0:
            self.Pinv['noiseadd'] = 1e-20
        yinlarge = yinlarge * (1 + np.random.normal(loc=0,scale=self.Pinv['noisemult'],size=len(yinlarge))) + np.random.normal(loc=0,scale=self.Pinv['noiseadd'],size=len(yinlarge))

        # assign to model variables
        self.stationids = indices
        self.F = F
        self.Khatlarge = Khatlarge
        self.yinlarge = yinlarge
        print('Computing K done')

    def inverseGmatrix(self):
        '''compute G matrix, depending on error covariances'''
        Sa = np.diag(self.Pinv['sigmaxa']*np.ones(self.P['nx']))
        Se = np.diag(self.Pinv['sigmaxe']*np.ones(self.P['nt']*len(self.Pinv['stations'])))

        # construct G matrix (Jacob, eq. 5.9)
        G = np.matmul(np.matmul(inv(np.matmul(np.matmul(self.Khatlarge.transpose(),inv(Se)),self.Khatlarge)+inv(Sa)),self.Khatlarge.transpose()),inv(Se))

        # Error covariance of initial vector (Jacob, eq. 5.10)
        Shat = inv(np.matmul(np.matmul(self.Khatlarge.transpose(),inv(Se)),self.Khatlarge) + inv(Sa))

        # assign to model variables
        self.Sa = Sa
        self.Se = Se
        self.G = G
        self.Shat = Shat
        print('Computing G done')

    def inverseSolution(self,xa=None):
        '''compute solution of inverse problem'''
        nx,nt = self.P['nx'],self.P['nt']
        # Compute best estimate
        if xa == None:
            xa = np.zeros(nx) # all zeros, no prior knowledge

        # compute initial vector based on yin, K, xa (Jacob, eq. 5.7)
        x = xa + np.matmul(self.G,self.yinlarge-np.matmul(self.Khatlarge,xa))

        # compute deviation between actual and recovered sources
        rmsdev = ( np.mean((x-self.P['E'])**2)  )**0.5
        # compute correlation between actual and recovered sources
        xcorr = np.corrcoef(x,self.P['E'])

        # assign to model variables
        self.Einv = x
        self.rmsdev = rmsdev
        self.Einvcorr = xcorr[0,1]
        print('Computing Inverse done')

    def inverseCost(self,x,xa):
        '''return the cost of a solution x relative to an initial guess xa'''
        return np.matmul(x-xa,np.matmul(inv(self.Sa),x-xa)) + np.matmul(self.yinlarge-np.matmul(self.Khatlarge,x),np.matmul(inv(self.Se),self.yinlarge-np.matmul(self.Khatlarge,x)))

    # ===================================
    # OUTPUT

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
