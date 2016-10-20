from numpy import *
from Class_Tracer_v2 import *
from scipy.optimize import *
import matplotlib.pyplot as plt
import matplotlib

class AdjoinedModel(object):
    def __init__(self,parameters,method='Upwind',initialvalue=None):
        self.P = dict.fromkeys(['xmax','dx','tmax','dt','u0','k','E_prior','E_true','stations','sigmaxa','sigmaxe','noisemult','noiseadd', 'precon','rerunning','accuracy'])
        self.P.update(parameters)
                
        # Initialize some quantities
        self.P['nx'] = np.int(self.P['xmax']/self.P['dx'])
        self.P['nt'] = np.int(self.P['tmax']/self.P['dt'])
        self.x = self.P['dx'] * np.arange(self.P['nx'])
        self.xvec  = np.arange(0,self.P['xmax'],self.P['dx'])
        
        self.Sa = np.diag(self.P['sigmaxa']*np.ones(self.P['nx']))
        self.Se = np.diag(self.P['sigmaxe']*np.ones(self.P['nt']))
        self.Sai = np.diag((1/self.P['sigmaxa'])*np.ones(self.P['nx']))
        self.Sei = np.diag((1/self.P['sigmaxe'])*np.ones(self.P['nt']))
        self.b       = self.Sa
        L_preco = sqrt(self.b)
        L_adj   = transpose(L_preco)
        self.L_adj = L_adj
        
        # Run true forward model
        Parameters_true = {'xmax':self.P['xmax'],'dx':self.P['dx'],'tmax':self.P['tmax'],'dt':self.P['dt'],'u0':self.P['u0'],'k':self.P['k'],'E':self.P['E_true']}
        m_true = TracerModel(Parameters_true,method=method,initialvalue=0)
        m_true.integrateModel()
        Obs_true_raw=m_true.results[:,stations]
        Obs_true=[]
        for i in range(0,len(Obs_true_raw)):
            Obs_true.append( Obs_true_raw[i]*(1+noisemult*np.random.uniform(low=-1,high=1,size=(len(Obs_true_raw[i]))))+noiseadd*np.random.uniform(low=-1,high=1,size=(len(Obs_true_raw[i]))))
        self.Obs_true=np.array(Obs_true)
        self.Obs_true_raw=Obs_true_raw
        self.Transport = m_true.Mtot2        
                  
        # Set initial value to zero, if not specified
        self.initialvalue = 0
        if initialvalue is not None:
            self.initialvalue = initialvalue

        # Set integration rule to Upwind + Euler, if not otherwise specified
        self.method = method

    def TestAdjoined(self,alpha,element):
        Cost_prior = AdjoinedModel.Cost(self,self.P['E_prior'])
        Derivative = AdjoinedModel.Adjoined(self,self.P['E_prior'],self.P['E_prior'])
        E_test=np.zeros(self.P['nx'])
        E0=1
        for j in range(0,self.P['nx']):
            x=np.int(j*self.P['dx'])
            self.xvec[j]=x
            if len(where(np.array(sources_guess)==x)[0])>0:
                E_test[j]=E0
        E_test[element]=E_test[element]+alpha
        Cost_test = AdjoinedModel.Cost(self,E_test)
        
        DE=np.zeros(self.P['nx'])-0.000000001
        E_new=np.zeros(self.P['nx'])
        E0=1
        for j in range(0,self.P['nx']):
            x=np.int(j*self.P['dx'])
            self.xvec[j]=x
            if len(where(np.array(sources_guess)==x)[0])>0:
                E_new[j]=E0
        print(Derivative)
        E_new=E_new+DE*Derivative
        Cost_new = AdjoinedModel.Cost(self,E_new)
        
        print('================================================')
        print('--- Testing ---')
        print('Prior cost function',Cost_prior)
        print('Derivative by calculation:',Derivative[element])
        print('Derivative by test:', (Cost_test-Cost_prior)/alpha)
        print('Cost function one time',Cost_new)
        print('Cost function change',Cost_new-Cost_prior)
        print('================================================')
        
    def AdjoinedModelling(self):
        L_preco = sqrt(self.b)
        L_adj   = transpose(L_preco)
        L_inv   = linalg.inv(L_preco)
        E_final = self.P['E_prior']
        
        print('================================================')
        print('--- Adjoined modelling started ---')
        
        E_final = self.P['E_prior']
        if self.P['precon']==1:
            for i in range(0,self.P['rerunning']):
                pstate, pderiv = state_to_precon(L_inv, L_adj, E_final, E_prior, Adjoined_priorint(E_final))
                state_opt=optimize.fmin_bfgs(Cost,pstate,Adjoined_priorint,gtol=self.P['accuracy'],disp=0)
                E_final = precon_to_state( L_preco, state_opt, E_prior )
                print('================================================')
                print('--- one run ended ---')
                print('================================================')
        elif self.P['precon']==0:
            for i in range(0,self.P['rerunning']):
                E_final=optimize.fmin_bfgs(Cost,E_final,Adjoined_priorint,gtol=self.P['accuracy'],disp=0)
                print('================================================')
                print('--- one run ended ---')
                print('================================================')
                
        self.E_final=E_final
        print('--- Adjoined modelling completed ---')
        print('================================================')
        
    def Plots(self):
        
        matplotlib.style.use('ggplot')
        fig=plt.figure(num=None, figsize=(7,3),dpi=150, facecolor='w', edgecolor='k') # little: 5,3, large: 9,3
        plt.scatter(self.P['stations'],np.zeros(len(self.P['stations']))+0.03,s=100,c='orange',alpha=1,zorder=15,edgecolor='k',linewidth=2,label='Measuring Stations')
        plt.plot(self.P['E_prior'],label='Prior',linewidth=4)
        plt.plot(self.P['E_true'],'dimgray',label='Actual',linewidth=2)
        plt.plot(self.E_final,label='Determined with preco',linewidth=4)
        plt.xlabel('Distance x',fontsize=15)
        plt.ylabel('Emission strength',fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlim([0,self.P['xmax']])
        plt.ylim([-0.2,1])
        plt.legend(fontsize=9)
        fig.tight_layout()
        plt.show()
        
        Parameters_iteration = {'xmax':self.P['xmax'],'dx':self.P['dx'],'tmax':self.P['tmax'],'dt':self.P['dt'],'u0':self.P['u0'],'k':self.P['k'],'E':self.E_final}
        m = TracerModel(Parameters_iteration,method=self.method,initialvalue=0)
        m.integrateModel()
        self.Obs_final=m.results[:,stations]
        Station = 0
        
        matplotlib.style.use('ggplot')
        fig=plt.figure(num=None, figsize=(7,3),dpi=150, facecolor='w', edgecolor='k') # little: 5,3, large: 9,3
        plt.plot(self.Obs_true_raw[:,Station],'dimgray',label='Actual',linewidth=2)
        plt.plot(self.Obs_true[:,Station],'orange',label='Actual measured',linewidth=2)
        plt.plot(self.Obs_final[:,Station],label='Determined',linewidth=4)
        plt.xlabel('Time',fontsize=15)
        plt.ylabel('Concentration',fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlim([0,self.P['tmax']])
        plt.legend(fontsize=9,loc='best')
        fig.tight_layout()
        plt.show()


# ------------------------------------------------------
# Three important functions
# ------------------------------------------------------

def Adjoined(x,x_prior):
    Parameters_iteration = {'xmax':m.P['xmax'],'dx':m.P['dx'],'tmax':m.P['tmax'],'dt':m.P['dt'],'u0':m.P['u0'],'k':m.P['k'],'E':x}
    M = TracerModel(Parameters_iteration,method='Upwind',initialvalue=0)
    M.integrateModel()
    Obs_iteration=M.results[:,stations]
    forcing = np.matmul(m.Sei,np.array(Obs_iteration)-np.array(m.Obs_true))  # (Hx-y)
    timevec=range(0,nt)
    C_adjoined = np.zeros(nx)
    E_adjoined = np.zeros(nx)
    for times in timevec[::-1]:
        C_adjoined[stations] = C_adjoined[stations] + forcing[times]
        C_adjoined = np.matmul(np.transpose(m.Transport),C_adjoined)
        E_adjoined = E_adjoined + C_adjoined*Dt
    derivative = 2*np.matmul(m.Sai,x-x_prior)+ 2*E_adjoined   
    m.derivative=derivative
    
def Cost(x):
    Parameters = {'xmax':m.P['xmax'],'dx':m.P['dx'],'tmax':m.P['tmax'],'dt':m.P['dt'],'u0':m.P['u0'],'k':m.P['k'],'E':x}
    M = TracerModel(Parameters,method=m.method,initialvalue=0)
    M.integrateModel()
    Obs=M.results[:,m.P['stations']]
    Cost=1/m.P['sigmaxe']*sum((np.array(m.Obs_true)-np.array(Obs))**2)+1/m.P['sigmaxa']*sum((np.array(x)-np.array(E_prior))**2)
    m.Costres=Cost
    return Cost

def Adjoined_priorint(x):
    '''
    adjoined model with given E_prior so that it has only 1 element
    '''
    Parameters_iteration = {'xmax':m.P['xmax'],'dx':m.P['dx'],'tmax':m.P['tmax'],'dt':m.P['dt'],'u0':m.P['u0'],'k':m.P['k'],'E':x}
    M = TracerModel(Parameters_iteration,method=m.method,initialvalue=0)
    M.integrateModel()
    Obs_iteration=M.results[:,m.P['stations']]
    forcing = np.matmul(m.Sei,np.array(Obs_iteration)-np.array(m.Obs_true))  # (Hx-y)
    C_adjoined = np.zeros(m.P['nx'])
    E_adjoined = np.zeros(m.P['nx'])
    timevec=range(0,m.P['nt'])
    for times in timevec[::-1]:
        C_adjoined[stations] = C_adjoined[m.P['stations']] + forcing[times]
        C_adjoined = np.matmul(np.transpose(m.Transport),C_adjoined)
        E_adjoined = E_adjoined + C_adjoined*m.P['dt']
    derivative = 2*np.matmul(m.Sai,x-m.P['E_prior'])+ 2*E_adjoined
    deriv=np.dot(m.L_adj,derivative)#???????????????????!
    print ('Cost function', Cost(x), 'Squared gradient',np.dot(deriv,deriv))
    m.deriv=deriv
    return deriv
    
def state_to_precon( L_inv, L_adj, state, state_apri, deriv ):
    '''
    Same function used in example code for preconditioning
    '''
    pstate = dot(L_inv,array(state)-array(state_apri))
    pderiv = dot(L_adj,array(deriv))
    return pstate, pderiv

def precon_to_state( L_precon, vals, state_apri ):
    '''
    Same function used in example code for preconditioning
    '''
    state = dot(L_precon, vals) + array(state_apri)
    return state