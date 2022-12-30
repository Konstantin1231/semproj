import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy import linalg as LA
from scipy.integrate import quad
'''
Parameters of true model
'''
names=['Monomials','Chebyshev','Hermite','Legendre','Laguerre']
model='non'
alpha_5=-1
alpha_4=0.7
alpha_3=1
alpha_2=0
alpha_1=-2
alpha_0=1
sigma_0=1
true=[alpha_0,alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,sigma_0]
true_OU=[-1,1]
true_bi=[-1,1,1]
Drift=np.polynomial.polynomial.Polynomial([0,1,0,-1])
Drift_= lambda x: (np.polynomial.polynomial.Polynomial([0,1,0,-1]).__call__(x))
'''Invariant measure'''
V= lambda x: -np.cos(x)+Drift.integ(1).__call__(x)
inv_mesure=lambda x: np.exp(V(x)/sigma_0)
mesure = lambda x: inv_mesure(x)/quad(inv_mesure,-np.inf,np.inf)[0]


"'SETTING"""
T=10000
degree=6
n_lines=4

"""SETTING"""






'''
Simulation
'''
def simulation(delta_time,J,X_0,str_):
    if str_=='OU':
        sim=[]
        sim.append(X_0)
        for _ in range(J):
            X_old = sim[len(sim) - 1]
            X_new = X_old + alpha_1 * X_old * delta_time + np.sqrt(2 * sigma_0 * delta_time) * norm.rvs()
            sim.append(X_new)
        return sim

    else:
        sim=[]
        sim.append(X_0)
        for _ in range(J):
            X_old=sim[len(sim)-1]
            X_new=X_old + Drift_.__call__(X_old)*delta_time+ np.sqrt(2*sigma_0*delta_time)*norm.rvs()
            sim.append(X_new)
        return sim



def Poly(X,J,max_degree,num_lines,class_):
    X=np.array( X[:J] )
    T = class_
    moments_diff=np.zeros((num_lines+1,max_degree+1))
    moments_2_diff=np.zeros((num_lines+1,max_degree+1))
    for ligne in range(1,num_lines+1):
        degree = np.zeros(num_lines+1)
        degree[ligne] = 1
        T_n_diff=T(degree).deriv(1)
        T_n_2_diff=T(degree).deriv(2)
        for _ in range(max_degree+1):
            degree = np.zeros(max_degree+1)
            degree[_] = 1
            T_k=T(degree)

            moments_diff[ligne,_]=np.sum(T_k.__call__(X)*T_n_diff.__call__(X))*(1/J)
            moments_2_diff[ligne,_]=np.sum(T_k.__call__(X)*T_n_2_diff.__call__(X))*(1/J)


    return moments_diff,moments_2_diff

''' 
Estimeteurs
'''


def Matrix(moments_diff,moments_2_diff):
    M=np.zeros([np.shape(moments_diff)[0],np.shape(moments_diff)[1]+1])
    for i in range(1,np.shape(moments_diff)[0]-1):
        M[i-1,:]=np.array(np.append(moments_diff[i,:],moments_2_diff[i,0]))
    M[np.shape(moments_diff)[0]-1,:]=np.array(np.append(np.zeros(np.shape(moments_diff)[1]),1))
    return M

def Limits(X):
    X_sorted=np.sort(X)
    lower_index=int(0.05*len(X))
    upper_index=int(0.95*len(X))
    return X_sorted[lower_index],X_sorted[upper_index]

def Error(lower, upper, f):
    return quad(f, lower, upper)[0]

def Moment(max_degree,X,J,delta_time):
    moments=np.ones(max_degree+1)
    V = np.array(X[0:J])
    for i in range(1,max_degree+1):
        moments[i]=(1/J) * np.sum(V)
        V=V*np.array(X[0:J])
    return moments

def Matrix_special(X,n_lignes,moments):
    M=np.zeros([n_lignes+1,2])
    M[0,:]=np.array([moments[1],0])
    for i in range(1,n_lignes):
        M[i,:]=np.array([moments[i+1],i*moments[i-1]])
    M[n_lignes,:]=np.array([0,1])
    return M

def Matrix_special_(X,n_lignes,moments):
    M=np.zeros([n_lignes+1,3])
    M[0,:]=np.array([moments[3],moments[1],0])
    for i in range(2,n_lignes+1):
        M[i-1,:]=np.array([moments[i+2],moments[i],(i-1)*moments[i-2]])
    M[n_lignes,:]=np.array([0,0,1])
    print(M)
    return M


'''
Calculating coef
'''
def finding_parameters(X,J,delta_time,M):
    A1 = np.array(X[1:J]) - np.array(X[0:J - 1])
    Q= (1 /(2* J*delta_time)) * np.dot(A1, A1)
    Y=np.zeros(np.shape(M)[0])
    Y[len(Y)-1]=Q
    reg = LinearRegression(fit_intercept=False).fit(M,Y)
    return reg.coef_



fig = plt.figure(figsize=(20, 10))
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)

x=[0,T]
ax1.plot(x,np.ones(len(x))*true_bi[2],color='red',label='True')
ax2.plot(x,np.ones(len(x))*true_bi[0],color='red',label='True')
ax3.plot(x,np.ones(len(x))*true_bi[1],color='red',label='True')

delta_time=0.001
J = int(T / delta_time)
X = simulation(delta_time, J, 0,model)
X_0=X[::10] # delta time 0.01
X_1=X[::50] # delta time 0.05
X_2=X #delta time 0.001
for X in [X_0, X_1, X_2]:
    alphas1 = []
    alphas3=[]
    sigmas = []
    J=len(X)
    delta_time=T/J
    xx = delta_time * np.array(range(int(100/delta_time),J,int(100/delta_time)))
    for j in range(int(100/delta_time),J,int(100/delta_time)):
        max_degree = n_lines+2
        moments = Moment(max_degree, X, j, delta_time)
        M = Matrix_special_(X, n_lines, moments)
        parameters=finding_parameters(X,j,delta_time,M)
        sigmas.append(parameters[2])
        alphas1.append(parameters[1])
        alphas3.append(parameters[0])

    ax1.plot(xx,sigmas)
    ax2.plot(xx, alphas3)
    ax3.plot(xx, alphas1, label=f'h={"{:.4f}".format(delta_time)}')

ax1.set(xlabel='X ', ylabel='Alpha_1')
ax1.grid()


ax2.set(xlabel='Time(T) ', ylabel='Alpha_3')
ax2.grid()


ax3.set(xlabel='Time(T) ', ylabel='Sigma')
ax3.grid()
ax3.legend(loc='center right',prop={'size': 10}, bbox_to_anchor=(1, 0.5))

plt.savefig(f'BI_same_simulations.jpeg')


plt.show()