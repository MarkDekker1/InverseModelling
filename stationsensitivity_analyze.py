# Inverse Modelling using Tracer Class

from Class_Tracer_v2 import *

res = np.load('sensitivities/sensitivity_stations.npy').all()


# power law fits
i = 1
fitK = np.polyfit(np.log(res['numstations'][i:]),np.log(res['timeK'][i:]),1)
i = 1
fitG = np.polyfit(np.log(res['numstations'][i:]),np.log(res['timeG'][i:]),1)

fig,ax = newfig2(0.5,0.8)
#fig,ax = plt.subplots(1)
ax.loglog(res['numstations'],res['timeK'],'.-',label='K matrix')
ax.loglog(res['numstations'],res['timeG'],'.-',label='G matrix')
ax.loglog(res['numstations'],np.exp(fitK[1])*res['numstations']**fitK[0],'k',lw=1,label=r'$N^{%.1f}$' % fitK[0])
ax.loglog(res['numstations'],np.exp(fitG[1])*res['numstations']**fitG[0],'grey',lw=1,label=r'$N^{%.1f}$' % fitG[0])
ax.set_xlim(res['numstations'].min(),res['numstations'].max())

ax.legend(loc=4)
ax.set_xlabel(r'number of stations $N$',labelpad=0)
ax.set_ylabel('runtime (s)',labelpad=0)

fig.tight_layout()
savefig(fig,'Figures/station_sensitivities')