import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
'''
Parameters of true model
'''
alpha=-1
sigma=1

''' 
Estimeteurs
'''
def sigma_(X,delta_time,J):
    A1=np.array(X[1:J]) - np.array(X[0:J-1])
    return np.sqrt((1/(J*delta_time)) * np.dot(A1,A1))

def alpha_(X,delta_time,J):
    V=np.array(X[0:J - 1])
    A1 =np.dot(np.array(X[1:J]) - V,V)
    A2= delta_time * np.dot(V,V)
    return A1/A2

'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new=X_old + (alpha)*X_old*delta_time+ sigma*np.sqrt(delta_time)*norm.rvs()
        sim.append(X_new)
    return sim

delta_time=0.1
J=80000
T=J*delta_time
X=simulation(delta_time,J,0)


'''
Main 
'''
alphas=[]
sigmas=[]
for j in range(200,J,100):
    sigmas.append(sigma_(X,delta_time,j))
    alphas.append(alpha_(X,delta_time,j))

fig=plt.figure(figsize=(15,5))
xx=delta_time*np.array(range(200,J,100))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
ax1.plot(xx,sigmas)
ax2.plot(xx,alphas)
ax1.set(xlabel='time(T) ', ylabel='Sigma')
ax1.grid()
ax2.set(xlabel='time(T) ', ylabel='Alpha')
ax2.grid()


plt.savefig('classical_exemple(-1,1).svg')


plt.show()
