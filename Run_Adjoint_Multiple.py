E_finals=[]
Obs_trues=[]
Obs_true_raws=[]
Obs_finals=[]
stations_vec=[]
amount=100
Errors_o=[]
Errors_e=[]
Max=[]
amountstations=4


for multi in range(0,amount):
    
    # ------------------------------------------------------
    # Define parameters
    # ------------------------------------------------------
    
    import numpy as np
    import time as T
    
    xmax            =   100
    u0              =   5
    tmax            =   10* xmax / u0
    k               =   0.1
    gamma           =   0.9
    Dx              =   xmax / 100
    Dt              =   gamma * Dx / u0
    E0              =   1
    sources         =   [10,30,60,90]
    #sources         =   np.random.randint(low=0,high=xmax,size=3)
    sources_guess   =   [100000]               #for determining prior
    stations        =   np.arange(0,xmax,1)     #all points have stations
    #stations        =   [20,40,60,80]
    stations        =   np.random.randint(low=0,high=xmax,size=amountstations)
    #stations        =   [15]
    #if amountstations==1:
    #    stations = [multi]
    stations_vec.append(stations)
    #stations        =   [19]
    #stations        =   np.arange(0,100,1)
    sigmaxa         =   2
    sigmaxe         =   0.004
    accuracy        =   1e-5
    noisemult       =   0
    noiseadd        =   0.004
    Preconditioning =   0
    Rerunning       =   5
    Offdiags        =   0
    BFGS            =   1
    Print           =   0
    Sa_vec          =   np.zeros(np.int(xmax/Dx))+sigmaxa#/10.
    #for i in range(0,13):
    #    Sa_vec[i]=sigmaxa
    #for j in range(45,55):
    #    Sa_vec[i]=sigmaxa
    #for i in [0,10,50]:
    #    Sa_vec[i]=sigmaxa
    for i in range(0,25):
        Sa_vec[i]=sigmaxa
    for i in range(40,70):
        Sa_vec[i]=sigmaxa
            
    # ------------------------------------------------------
    # Emission vectors
    # ------------------------------------------------------
    
    nx          =   np.int(xmax/Dx)
    nt          =   np.int(tmax/Dt)
    xvec        =   np.arange(0,xmax,Dx)
    E_prior     =   np.zeros(nx)
    E_true      =   np.zeros(nx)
    for j in range(0,nx):
        x       =   j*Dx
        xvec[j] =   x
        if len(np.where(np.array(sources_guess)==x)[0])>0:
            E_prior[j]=E0
        if len(np.where(np.array(sources)==x)[0])>0:
            E_true[j]=E0
            
    # ------------------------------------------------------
    # Define dictionary for class
    # ------------------------------------------------------
            
    Parameters = {
        'xmax':xmax,
        'dx':Dx,
        'tmax':tmax,
        'dt':Dt,
        'u0':u0,
        'k':k,
        'E_prior':E_prior,
        'E_true':E_true,
        'stations':stations,
        'sigmaxa':sigmaxa,
        'sigmaxe':sigmaxe,
        'noisemult':noisemult,
        'noiseadd':noiseadd,
        'precon':Preconditioning,
        'rerunning':Rerunning,
        'accuracy':accuracy,
        'Offdiags':Offdiags,
        'BFGS':BFGS,
        'Sa_vec':Sa_vec
        }
    
    m = AdjointModel(Parameters,method='Upwind',initialvalue=0)
    
    m.AdjointModelling()
    
    E_finals.append(m.E_final)
    Obs_trues.append(m.Obs_true)
    Obs_true_raws.append(m.Obs_true_raw)
    Obs_finals.append(m.Obs_final)
    Errors_o.append(sum(m.Obs_final-m.Obs_true)**2)
    Errors_e.append(sum(m.E_final-E_true)**2)
    Max.append(max(m.E_final))
    print(multi)
#%%
# ------------------------------------------------------
# Save into right matrices
# ------------------------------------------------------
amountmaxmin=1
#%%
Errorse_1=np.array(Errors_e)
Errorso_1=np.array(Errors_o)
E_finals_1=np.array(E_finals)
E_Obs_finals_1=np.array(Obs_finals)
E_Obs_true_raws_1=np.array(Obs_true_raws)
E_Obs_trues_1=np.array(Obs_trues)
stations_vec_1=stations_vec
Max_ind_1=where(Max==max(Max))
#%%
Best_ind_1=np.array(Errorso_1).argsort()[:amountmaxmin][::-1]
Worst_ind_1=np.array(Errorso_1).argsort()[-amountmaxmin:][::-1]
#%%
Errorse_4=np.array(Errors_e)
Errorso_4=np.array(Errors_o)
E_finals_4=np.array(E_finals)
E_Obs_finals_4=np.array(Obs_finals)
E_Obs_true_raws_4=np.array(Obs_true_raws)
E_Obs_trues_4=np.array(Obs_trues)
stations_vec_4=stations_vec
Max_ind_4=where(Max==max(Max))
#%%
Best_ind_4=np.array(Errorso_4).argsort()[:amountmaxmin][::-1]
Worst_ind_4=np.array(Errorso_4).argsort()[-amountmaxmin:][::-1]
#%%
Errorse_6=np.array(Errors_e)
Errorso_6=np.array(Errors_o)
E_finals_6=np.array(E_finals)
E_Obs_finals_6=np.array(Obs_finals)
E_Obs_true_raws_6=np.array(Obs_true_raws)
E_Obs_trues_6=np.array(Obs_trues)
stations_vec_6=stations_vec
Max_ind_6=where(Max==max(Max))
#%%
Best_ind_6=np.array(Errorso_6).argsort()[:amountmaxmin][::-1]
Worst_ind_6=np.array(Errorso_6).argsort()[-amountmaxmin:][::-1]
    
#%% Upwinds
Errorse_1_u=np.array(Errors_e)
Errorso_1_u=np.array(Errors_o)
E_finals_1_u=np.array(E_finals)
E_Obs_finals_1_u=np.array(Obs_finals)
E_Obs_true_raws_1_u=np.array(Obs_true_raws)
E_Obs_trues_1_u=np.array(Obs_trues)
stations_vec_1_u=stations_vec
Max_ind_1_u=where(Max==max(Max))
#%%
Best_ind_1_u=np.array(Errorso_1).argsort()[:amountmaxmin][::-1]
Worst_ind_1_u=np.array(Errorso_1).argsort()[-amountmaxmin:][::-1]
#%%
Errorse_4_u=np.array(Errors_e)
Errorso_4_u=np.array(Errors_o)
E_finals_4_u=np.array(E_finals)
E_Obs_finals_4_u=np.array(Obs_finals)
E_Obs_true_raws_4_u=np.array(Obs_true_raws)
E_Obs_trues_4_u=np.array(Obs_trues)
stations_vec_4_u=stations_vec
Max_ind_4_u=where(Max==max(Max))
#%%
Best_ind_4_u=np.array(Errorso_6).argsort()[:amountmaxmin][::-1]
Worst_ind_4_u=np.array(Errorso_6).argsort()[-amountmaxmin:][::-1]
#%%
Errorse_6_u=np.array(Errors_e)
Errorso_6_u=np.array(Errors_o)
E_finals_6_u=np.array(E_finals)
E_Obs_finals_6_u=np.array(Obs_finals)
E_Obs_true_raws_6_u=np.array(Obs_true_raws)
E_Obs_trues_6_u=np.array(Obs_trues)
stations_vec_6_u=stations_vec
Max_ind_6_u=where(Max==max(Max))
#%%
Best_ind_6_u=np.array(Errorso_6).argsort()[:amountmaxmin][::-1]
Worst_ind_6_u=np.array(Errorso_6).argsort()[-amountmaxmin:][::-1]
    
#%%
# ------------------------------------------------------
# Plots
# ------------------------------------------------------
tocheck=44
if tocheck==1:
    Finals=E_finals_1
    stations_vec=stations_vec_1
    Worst=Worst_ind_1
    Best=Best_ind_1
    Maxi=Max_ind_1
if tocheck==4:
    Finals=E_finals_4
    stations_vec=stations_vec_4
    Worst=Worst_ind_4
    Best=Best_ind_4
    Maxi=Max_ind_4
if tocheck==6:
    Finals=E_finals_6
    stations_vec=stations_vec_6
    Worst=Worst_ind_6
    Best=Best_ind_6
    Maxi=Max_ind_6
if tocheck==11:
    Finals=E_finals_1_u
    stations_vec=stations_vec_1_u
    Worst=Worst_ind_1_u
    Best=Best_ind_1_u
    Maxi=Max_ind_1_u
if tocheck==44:
    Finals=E_finals_4_u
    stations_vec=stations_vec_4_u
    Worst=Worst_ind_4_u
    Best=Best_ind_4_u
    Maxi=Max_ind_4_u
if tocheck==66:
    Finals=E_finals_6_u
    stations_vec=stations_vec_6_u
    Worst=Worst_ind_6_u
    Best=Best_ind_6_u
    Maxi=Max_ind_6_u

fontsize1=20
matplotlib.style.use('ggplot')
plt.style.use('seaborn-white')
fig=plt.figure(num=None, figsize=(12,3),dpi=600, facecolor='w', edgecolor='k') 

for i in range(0,amount):
#for i in Best:
#for i in Best:
    a=plt.plot(xvec+1,np.transpose(Finals[i]),label='Determined with preco',linewidth=4)
    #plt.scatter(stations_vec[i],np.zeros(len(stations_vec[i]))+0.02,s=100,c=a[0].get_color(),alpha=1,zorder=15,edgecolor='k',linewidth=2,label='Measuring Stations')

plt.plot(E_true,'dimgray',label='Actual',linewidth=2,zorder=-1)
#plt.plot([0,0],[0.5,0.5],'k',linewidth=5,zorder=15)
plt.xlabel('Distance x',fontsize=fontsize1)
plt.ylabel('Emission strength',fontsize=fontsize1)
plt.tick_params(axis='both', which='major', labelsize=fontsize1)
plt.xlim([0,xmax])
plt.ylim([-0.2,1])
#plt.legend(loc='best')
fig.tight_layout()
plt.show()

#%%
# ------------------------------------------------------
# PlotsBest
# ------------------------------------------------------


fontsize1=20
#matplotlib.style.use('ggplot')
plt.style.use('seaborn-white')
fig=plt.figure(num=None, figsize=(8,5),dpi=550, facecolor='w', edgecolor='k') 
a=plt.plot(xvec+1,np.transpose(E_finals_1_u[Best_ind_1_u]),label='Determined with preco',linewidth=4)
#plt.scatter(stations_vec_1[Best_ind_1],np.zeros(len(stations_vec_1[Best_ind_1]))+0.02,s=100,c=a[0].get_color(),alpha=1,zorder=15,edgecolor='k',linewidth=2,label='Measuring Stations')

b=plt.plot(xvec+1,np.transpose(E_finals_4_u[Best_ind_4_u]),label='Determined with preco',linewidth=4)
#plt.scatter(stations_vec_4[Best_ind_4],np.zeros(len(stations_vec_4[Best_ind_4]))+0.02,s=100,c=b[0].get_color(),alpha=1,zorder=15,edgecolor='k',linewidth=2,label='Measuring Stations')

b=plt.plot(xvec+1,np.transpose(E_finals_6_u[Best_ind_6_u]),label='Determined with preco',linewidth=4)
#plt.scatter(stations_vec_6[Best_ind_6],np.zeros(len(stations_vec_6[Best_ind_6]))+0.02,s=100,c=b[0].get_color(),alpha=1,zorder=15,edgecolor='k',linewidth=2,label='Measuring Stations')

plt.plot(E_true,'dimgray',label='Actual',linewidth=2,zorder=-1)
plt.xlabel('Distance x',fontsize=fontsize1)
plt.ylabel('Emission strength',fontsize=fontsize1)
plt.tick_params(axis='both', which='major', labelsize=fontsize1)
plt.xlim([0,xmax])
plt.ylim([-0.2,1])
#plt.legend(loc='best')
fig.tight_layout()
plt.show()

#%%
# ------------------------------------------------------
# Plots worst
# ------------------------------------------------------


fontsize1=20
#matplotlib.style.use('ggplot')
plt.style.use('seaborn-white')
fig=plt.figure(num=None, figsize=(8,5),dpi=550, facecolor='w', edgecolor='k') 
a=plt.plot(xvec+1,np.transpose(E_finals_1[Worst_ind_1]),label='Determined with preco',linewidth=4)
#plt.scatter(stations_vec_1[Worst_ind_1],np.zeros(len(stations_vec_1[Worst_ind_1]))+0.02,s=100,c=a[0].get_color(),alpha=1,zorder=15,edgecolor='k',linewidth=2,label='Measuring Stations')

b=plt.plot(xvec+1,np.transpose(E_finals_4[Worst_ind_4]),label='Determined with preco',linewidth=4)
#plt.scatter(stations_vec_4[Worst_ind_4],np.zeros(len(stations_vec_4[Worst_ind_4]))+0.02,s=100,c=b[0].get_color(),alpha=1,zorder=15,edgecolor='k',linewidth=2,label='Measuring Stations')

b=plt.plot(xvec+1,np.transpose(E_finals_6[Worst_ind_6]),label='Determined with preco',linewidth=4)
#plt.scatter(stations_vec_6[Worst_ind_6],np.zeros(len(stations_vec_6[Worst_ind_6]))+0.02,s=100,c=b[0].get_color(),alpha=1,zorder=15,edgecolor='k',linewidth=2,label='Measuring Stations')

plt.plot(E_true,'dimgray',label='Actual',linewidth=2,zorder=-1)
plt.xlabel('Distance x',fontsize=fontsize1)
plt.ylabel('Emission strength',fontsize=fontsize1)
plt.tick_params(axis='both', which='major', labelsize=fontsize1)
plt.xlim([0,xmax])
plt.ylim([-0.2,1])
#plt.legend(loc='best')
fig.tight_layout()
plt.show()