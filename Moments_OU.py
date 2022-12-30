import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
exact_moments=[1,0,1,0,3,0,15,0,105,0]
alpha=-1
sigma=1
'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new=X_old + alpha*X_old*delta_time+ np.sqrt(2*sigma*delta_time)*norm.rvs()
        sim.append(X_new)
    return sim

''' 
Estimeteurs
'''
def Moment(max_degree,X,J,delta_time):
    moments=np.ones(max_degree+1)
    V = np.array(X[0:J])
    for i in range(1,max_degree+1):
        moments[i]=(1/J) * np.sum(V)
        V=V*np.array(X[0:J])
    return moments



delta_time=0.01

T=15000
J=int(T/delta_time)
X=simulation(delta_time,J,0)

fig = plt.figure(figsize=(15, 5))
xx=np.array(range(2,9))
ax1=fig.add_subplot(111)
xx = delta_time * np.array(range(70000,J,1000))

error=np.zeros(shape=(len(range(70000, J, 1000)),10))
for j in range(70000, J, 1000):
    moments = Moment(9,X,j,delta_time)
    print(np.abs(exact_moments-moments))
    error[int((j-70000)/1000),:]=np.abs(exact_moments-moments)
for _ in range(10):
    ax1.plot(np.log(xx), np.log(error[:,_]), label=f'{_}')


ax1.set(xlabel='time(T) ', ylabel='Error')
ax1.grid()
ax1.legend(title="Degree:",loc='center right', bbox_to_anchor=(1, 0.5))
plt.savefig(f'Moment_estimation_OU({alpha},{sigma}).svg')

plt.show()