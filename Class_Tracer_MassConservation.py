#Check mass conservation
C=m1.results
Mass_result=np.zeros(len(C))
tvec=np.arange(0,tmax,Dt)

for i in range(0,len(C)):
    Mass_result[i]=sum(C[i])*Dx
    
Mass_theory=np.zeros(len(C))
for i in range(1,len(C)):
    Mass_theory[i]=Dt*sum(E)*Dx-Dt*k*Mass_theory[i-1]+Mass_theory[i-1]

plt.figure(num=None, figsize=(10,4),dpi=150, facecolor='w', edgecolor='k')
plt.plot(tvec,Mass_theory, 'k-',linewidth=2)
plt.plot(tvec,Mass_result, 'r-',linewidth=2)
plt.ylabel(r'Mass',fontsize=15)
plt.xlabel('Time [s]',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)

plt.figure(num=None, figsize=(10,4),dpi=150, facecolor='w', edgecolor='k')
plt.plot(tvec,Mass_theory-Mass_result, 'b-',linewidth=2)
plt.ylabel(r'Mass',fontsize=15)
plt.xlabel('Time [s]',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=10)