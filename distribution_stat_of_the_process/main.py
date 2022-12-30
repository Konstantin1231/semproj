import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
alpha=-1
sigma=1
alpha_0=0.1
X_0=0
'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new=X_old +-X_old*delta_time+ np.sqrt(2*sigma*delta_time)*norm.rvs()
        sim.append(X_new)
    return sim

def simulation_(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new=X_old +(X_old-X_old**3)*delta_time+ np.sqrt(2*sigma*delta_time)*norm.rvs()
        sim.append(X_new)
    return sim

def Moment(max_degree,X,J,delta_time):
    moments=np.ones(max_degree+1)
    V = np.array(X[0:J])
    for i in range(1,max_degree+1):
        moments[i]=(1/J) * np.sum(V)
        V=V*np.array(X[0:J])
    return moments




'''
Calculating coef
'''



T=3000
delta_time=0.01
J=int(T/delta_time)
max_degree=7
X=simulation(delta_time,J,X_0)
Y=simulation_(delta_time,J,X_0)
moments_X=Moment(max_degree,X,J,delta_time)
moments_Y=Moment(max_degree,Y,J,delta_time)

fig = plt.figure(figsize=(15, 10))
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
ax4=fig.add_subplot(224)


ax1.hist(X,bins=30)
ax1.set_title('OU')
ax2.hist(Y,bins=30)
ax2.set_title('Bistable')
ax3.boxplot([X,Y])
ax3.set_xticklabels(['OU','Bistable'])
ax4.scatter([0,1,2,3,4,5,6,7],moments_X,label='OU')
ax4.scatter([0,1,2,3,4,5,6,7],moments_Y,label='Bistable')
ax4.plot([0,7],[0,0],color='red')

ax1.set(xlabel='$X_{t}$ ')
ax2.set(xlabel='$Y_{t}$ ')
ax4.set(xlabel='Degree: n', ylabel='$M_{n}$')

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax4.legend(loc='center right',prop={'size': 9}, bbox_to_anchor=(1.1, 0.5))

plt.savefig(f'STATISTIC.jpeg')



plt.show()
