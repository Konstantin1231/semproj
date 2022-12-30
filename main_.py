import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
'''
Parameters of true model
'''
alpha_1=1
alpha_3=-1
sigma=1
''' 
Estimeteurs
'''
def Moment(max_degree,X,j):
    moments=np.ones(max_degree+1)
    V = np.array(X[0:j])
    for i in range(1,max_degree+1):
        moments[i]=(1/j) * np.sum(V)
        V=V*np.array(X[0:j])
    return moments
def stoc_int(max_degree,X,J,delta_time):
    moments = np.ones(max_degree + 1)
    dif=np.array(X[1:J]) - np.array(X[0:J-1])

    V=np.array(X[0:J - 1])
    for i in range(1, max_degree + 1):
        moments[i] =  (1 / (J*delta_time)) * np.dot(V,dif)
        V = V * np.array(X[0:J - 1])

    return moments

def sigma_(X,delta_time,J):
    A1=np.array(X[1:J]) - np.array(X[0:J-1])
    return np.sqrt((1/(J*delta_time)) * np.dot(A1,A1))

def alpha_(X,delta_time,J):
    M=Moment(7,X,J)
    B=stoc_int(4,X,J,delta_time)
    print(B)
    alpha_1=(B[1]*M[6]-B[3]*M[4])/(M[2]*M[6]-M[4]**2)
    alpha_3=-(B[1]*M[4]-B[3]*M[2])/(M[2]*M[6]-M[4]**2)
    return [alpha_1,alpha_3]

'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new=X_old +(alpha_3*(X_old**3) + alpha_1*X_old)*delta_time+ sigma*np.sqrt(delta_time)*norm.rvs()
        sim.append(X_new)
    return sim

delta_time=0.01
J=200000
T=J*delta_time
X=simulation(delta_time,J,0)


'''
Main 
'''
alphas_1=[]
alphas_3=[]
sigmas=[]
for j in range(500,J,500):
    sigmas.append(sigma_(X,delta_time,j))
    alphas_1.append(alpha_(X,delta_time,j)[0])
    alphas_3.append(alpha_(X, delta_time, j)[1])
fig=plt.figure(figsize=(15,5))
xx=delta_time*np.array(range(500,J,500))
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)
ax1.plot(xx,sigmas)
ax2.plot(xx,alphas_1)
ax3.plot(xx,alphas_3)
ax1.set(xlabel='time(T) ', ylabel='Sigma')
ax1.grid()
ax2.set(xlabel='time(T) ', ylabel='Alpha_1')
ax2.grid()
ax3.set(xlabel='time(T) ', ylabel='Alpha_3')
ax3.grid()



plt.savefig(f'classical_exemple({alpha_3},{alpha_1},{sigma}).svg')


plt.show()