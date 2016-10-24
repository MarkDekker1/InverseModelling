from numpy import *
import numpy as np
#from Class_Tracer_v2 import *
from scipy.optimize import *
import matplotlib.pyplot as plt
import matplotlib

class AdjointModel(object):
    def __init__(self,parameters,method='Upwind',initialvalue=None):
        import numpy as np
        self.P = dict.fromkeys(['xmax','dx','tmax','dt','u0','k','E_prior','E_true','stations','sigmaxa','sigmaxe','noisemult','noiseadd', 'precon','rerunning','accuracy','Offdiags', 'BFGS','Sa_vec'])
        self.P.update(parameters)
                
        # Initialize some quantities
        self.P['nx'] = np.int(self.P['xmax']/self.P['dx'])
        self.P['nt'] = np.int(self.P['tmax']/self.P['dt'])
        self.x = self.P['dx'] * np.arange(self.P['nx'])
        self.xvec  = np.arange(0,self.P['xmax'],self.P['dx'])
        
        Sa = np.diag(self.P['sigmaxa']*np.ones(self.P['nx']))
        Sa = np.diag(self.P['Sa_vec'])
        Se = np.diag(self.P['sigmaxe']*np.ones(self.P['nt']))
        
        if self.P['Offdiags']==1:
            for i in range(0,len(Sa)):
                for j in range(0,len(Sa[i])):
                    if np.abs(i-j)<5 and i!=j:
                        Sa[i,j]=self.P['sigmaxa']*exp(-(np.abs(i-j)/5.))
            for i in range(0,len(Se)):
                for j in range(0,len(Se[i])):
                    if np.abs(i-j)<5 and i!=j:
                        Se[i,j]=self.P['sigmaxe']*exp(-(np.abs(i-j)/5.))
            self.Sai = linalg.inv(Sa)
            self.Sei = linalg.inv(Se)
        else:
            self.Sai = linalg.inv(Sa)#np.diag((1/self.P['sigmaxa'])*np.ones(self.P['nx']))
            self.Sei = linalg.inv(Se)#np.diag((1/self.P['sigmaxe'])*np.ones(self.P['nt']))
            
            
        self.Sa=Sa
        self.Se=Se
                
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
        
        # Preconditioning
        if self.P['precon']==1:
            self.B=self.Sa
            self.values,self.vectors = np.linalg.eig(self.B)
            self.V=np.matmul(np.transpose(self.vectors),np.sqrt(np.diag(self.values)))
            self.Vi=np.linalg.inv(self.V)
            
        self.b          = self.Sa
        L_preco         = np.sqrt(self.b)
        L_adj           = np.transpose(L_preco)
        self.L_adj      = L_adj
                  
        # Set initial value to zero, if not specified
        self.initialvalue = 0
        if initialvalue is not None:
            self.initialvalue = initialvalue

        # Set integration rule to Upwind + Euler, if not otherwise specified
        self.method = method

    def TestAdjoint(self,alpha,element):
        Cost_prior = Cost(self.P['E_prior'])
        Derivative = Adjoint(self.P['E_prior'],self.P['E_prior'])
        E_test=np.zeros(self.P['nx'])
        E0=1
        for j in range(0,self.P['nx']):
            x=np.int(j*self.P['dx'])
            self.xvec[j]=x
            if len(where(np.array(sources_guess)==x)[0])>0:
                E_test[j]=E0
        E_test[element]=E_test[element]+alpha
        Cost_test = Cost(E_test)
        
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
        Cost_new = Cost(E_new)
        
        print('================================================')
        print('--- Testing ---')
        print('Prior cost function',Cost_prior)
        print('Derivative by calculation:',Derivative[element])
        print('Derivative by test:', (Cost_test-Cost_prior)/alpha)
        print('Cost function one time',Cost_new)
        print('Cost function change',Cost_new-Cost_prior)
        print('================================================')
        
    def AdjointModelling(self):
        L_preco = np.sqrt(self.b)
        L_adj   = np.transpose(L_preco)
        L_inv   = np.linalg.inv(L_preco)
        E_final = self.P['E_prior']
        
        if Print==1:    
            print('================================================')
            print('--- Adjoint modelling started ---')
            print('================================================')
        
        E_final = self.P['E_prior']
        if self.P['BFGS']==1:
            if self.P['precon']==1:
                for i in range(0,self.P['rerunning']):
                    #pstate, pderiv = state_to_precon(L_inv, L_adj, matmul(self.V,E_final, matmul(self.V,E_prior), np.matmul(self.V,Adjoint_priorint(E_final)))
                    self.precon_prior=matmul(self.V,E_final)
                    state_opt=optimize.fmin_bfgs(Cost_precon,self.precon_prior,Adjoint_precon,gtol=self.P['accuracy'],disp=0)
                    #E_final = np.matmul(self.Vi,state_opt)
                    E_final=state_opt
                    if Print==1:
                        print('================================================')
                        print('--- one run ended ---')
                        print('================================================')
            elif self.P['precon']==0:
                for i in range(0,self.P['rerunning']):
                    E_final=optimize.fmin_bfgs(Cost,E_final,Adjoint_priorint,gtol=self.P['accuracy'],disp=0)
                    if Print==1:
                        print('================================================')
                        print('--- one run ended ---')
                        print('================================================')
        if self.P['BFGS']==0:
            if self.P['precon']==1:
                for i in range(0,self.P['rerunning']):
                    self.precon_prior=matmul(self.V,E_final)
                    state_opt=optimize.fmin_cg(Cost_precon,self.precon_prior,Adjoint_precon,gtol=self.P['accuracy'],disp=0)
                    #E_final = np.matmul(self.Vi,state_opt)
                    E_final=state_opt
                    
                    
                    #pstate, pderiv = state_to_precon(L_inv, L_adj, E_final, E_prior, Adjoint_priorint(E_final))
                    #state_opt=optimize.fmin_cg(Cost,pstate,Adjoint_priorint,gtol=self.P['accuracy'],disp=0)
                    #E_final = precon_to_state( L_preco, state_opt, E_prior )
                    if Print==1:
                        print('================================================')
                        print('--- one run ended ---')
                        print('================================================')
            elif self.P['precon']==0:
                for i in range(0,self.P['rerunning']):
                    E_final=optimize.fmin_cg(Cost,E_final,Adjoint_priorint,gtol=self.P['accuracy'],disp=0)
                    if Print==1:
                        print('================================================')
                        print('--- one run ended ---')
                        print('================================================')
                
        self.E_final=E_final
        
        Parameters_iteration = {'xmax':self.P['xmax'],'dx':self.P['dx'],'tmax':self.P['tmax'],'dt':self.P['dt'],'u0':self.P['u0'],'k':self.P['k'],'E':self.E_final}
        m = TracerModel(Parameters_iteration,method=self.method,initialvalue=0)
        m.integrateModel()
        self.Obs_final=m.results[:,stations]
        if Print==1:
            print('--- Adjoint modelling completed ---')
            print('================================================')
        
    def Plots(self,Station):
        
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
        plt.ylim([0,1])
        plt.legend(fontsize=9)
        fig.tight_layout()
        plt.show()
        
        Parameters_iteration = {'xmax':self.P['xmax'],'dx':self.P['dx'],'tmax':self.P['tmax'],'dt':self.P['dt'],'u0':self.P['u0'],'k':self.P['k'],'E':self.E_final}
        m = TracerModel(Parameters_iteration,method=self.method,initialvalue=0)
        m.integrateModel()
        self.Obs_final=m.results[:,stations]
        
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
# Some important functions
# ------------------------------------------------------

def Adjoint(x,x_prior):
    Parameters_iteration = {'xmax':m.P['xmax'],'dx':m.P['dx'],'tmax':m.P['tmax'],'dt':m.P['dt'],'u0':m.P['u0'],'k':m.P['k'],'E':x}
    M = TracerModel(Parameters_iteration,method=m.method,initialvalue=0)
    M.integrateModel()
    Obs_iteration=M.results[:,stations]
    forcing = np.matmul(m.Sei,np.array(Obs_iteration)-np.array(m.Obs_true))  # (Hx-y)
    timevec=range(0,nt)
    C_Adjoint = np.zeros(nx)
    E_Adjoint = np.zeros(nx)
    for times in timevec[::-1]:
        C_Adjoint[stations] = C_Adjoint[stations] + forcing[times]*m.P['dt']
        C_Adjoint = np.matmul(np.transpose(m.Transport),C_Adjoint)
        E_Adjoint = E_Adjoint + C_Adjoint
    derivative = 2*np.matmul(m.Sai,x-x_prior)+ 2*E_Adjoint   
    m.derivative=derivative
    return derivative
    
def Cost(x):
    Parameters = {'xmax':m.P['xmax'],'dx':m.P['dx'],'tmax':m.P['tmax'],'dt':m.P['dt'],'u0':m.P['u0'],'k':m.P['k'],'E':x}
    M = TracerModel(Parameters,method=m.method,initialvalue=0)
    M.integrateModel()
    Obs=M.results[:,m.P['stations']]
    Costf=1/m.P['sigmaxe']*sum((np.array(m.Obs_true)-np.array(Obs))**2)+np.matmul(np.array(x)-np.array(E_prior),np.matmul(m.Sai,np.array(x)-np.array(E_prior)))
    m.Costres=Costf
    return Costf

def Adjoint_priorint(x):
    '''
    Adjoint model with given E_prior so that it has only 1 element
    '''
    Parameters_iteration = {'xmax':m.P['xmax'],'dx':m.P['dx'],'tmax':m.P['tmax'],'dt':m.P['dt'],'u0':m.P['u0'],'k':m.P['k'],'E':x}
    M = TracerModel(Parameters_iteration,method=m.method,initialvalue=0)
    M.integrateModel()
    Obs_iteration=M.results[:,m.P['stations']]
    forcing = np.matmul(m.Sei,np.array(Obs_iteration)-np.array(m.Obs_true))  # (Hx-y)
    C_Adjoint = np.zeros(m.P['nx'])
    E_Adjoint = np.zeros(m.P['nx'])
    timevec=range(0,m.P['nt'])
    for times in timevec[::-1]:
        C_Adjoint[stations] = C_Adjoint[m.P['stations']] + forcing[times]*m.P['dt']
        C_Adjoint = np.matmul(np.transpose(m.Transport),C_Adjoint)
        E_Adjoint = E_Adjoint + C_Adjoint#*m.P['dt']
    derivative = 2*np.matmul(m.Sai,x-m.P['E_prior'])+ 2*E_Adjoint
    if Print==1:
        print ('Cost function', Cost(x), 'Squared gradient',np.dot(derivative,derivative))
    m.deriv=derivative
    m.derivative=np.dot(derivative,derivative)
    m.cost=Cost(x)
    return derivative
    
def Adjoint_precon(x):
    '''
    Adjoint model with given E_prior so that it has only 1 element
    '''
    Parameters_iteration = {'xmax':m.P['xmax'],'dx':m.P['dx'],'tmax':m.P['tmax'],'dt':m.P['dt'],'u0':m.P['u0'],'k':m.P['k'],'E':np.matmul(m.Vi,x)}
    M = TracerModel(Parameters_iteration,method=m.method,initialvalue=0)
    M.integrateModel()
    Obs_iteration=M.results[:,m.P['stations']]
    forcing = np.matmul(m.Sei,np.array(Obs_iteration)-np.array(m.Obs_true))  # (Hx-y)
    C_Adjoint = np.zeros(m.P['nx'])
    E_Adjoint = np.zeros(m.P['nx'])
    timevec=range(0,m.P['nt'])
    for times in timevec[::-1]:
        C_Adjoint[stations] = C_Adjoint[m.P['stations']] + forcing[times]*m.P['dt']
        C_Adjoint = np.matmul(np.transpose(m.Transport),C_Adjoint)
        E_Adjoint = E_Adjoint + C_Adjoint#*m.P['dt']
    derivative = 2*(x-np.matmul(m.V,m.P['E_prior']))+ 2*E_Adjoint
    if Print==1:
        print ('Cost function', Cost(x), 'Squared gradient',np.dot(derivative,derivative))
    derivative=np.matmul(np.linalg.inv(m.Vi),derivative)
    m.deriv=derivative
    m.derivative=np.dot(derivative,derivative)
    m.cost=Cost(x)
    return derivative
    
def Cost_precon(x):
    Parameters = {'xmax':m.P['xmax'],'dx':m.P['dx'],'tmax':m.P['tmax'],'dt':m.P['dt'],'u0':m.P['u0'],'k':m.P['k'],'E':np.matmul(m.Vi,x)}
    M = TracerModel(Parameters,method=m.method,initialvalue=0)
    M.integrateModel()
    Obs=M.results[:,m.P['stations']]
    Costf=1/m.P['sigmaxe']*sum((np.array(m.Obs_true)-np.array(Obs))**2)+sum((np.matmul(m.V,x)-np.matmul(m.V,np.array(E_prior)))**2)
    m.Costres=Costf
    return Costf
    
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