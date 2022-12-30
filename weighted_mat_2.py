import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy import linalg as LA
from scipy.integrate import quad
'''
Parameters of true model
'''

model='non'
alpha_5=-1
alpha_4=0.7
alpha_3=3
alpha_2=0
alpha_1=-2
alpha_0=1
sigma_0=3
sigma_1=0
sigma_2=0
true=[alpha_0,alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,sigma_0,sigma_1,sigma_2]
Diffusion=np.polynomial.polynomial.Polynomial([sigma_0,sigma_1,sigma_2])
Drift=np.polynomial.polynomial.Polynomial([alpha_0,alpha_1,alpha_2,alpha_3,alpha_4,alpha_5])
Drift_= lambda x: (np.polynomial.polynomial.Polynomial([alpha_0,alpha_1,alpha_2,alpha_3,alpha_4,alpha_5]).__call__(x))
'''Invariant measure'''
V= lambda x: Drift.integ(1).__call__(x)
inv_mesure=lambda x: np.exp(V(x)/sigma_0)
mesure = lambda x: inv_mesure(x)/quad(inv_mesure,-np.inf,np.inf)[0]
"'SETTING"""
T=5000
delta_time=0.01
#np.polynomial.chebyshev.Chebyshev
#np.polynomial.polynomial.Polynomial
#np.polynomial.hermite.Hermite
#numpy.polynomial.legendre.Legendre
#np.polynomial.laguerre.Laguerre
degree=4
n_lines=4
"""SETTING"""

"""advance setting if model is known"""
drift_model=[1,1,1,1,1,1]
diffusion_model=[1,1,1]


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
            X_new=X_old + (Drift_.__call__(X_old)+np.sin(3*X_old))*delta_time+ np.sqrt(2*sigma_0*delta_time)*norm.rvs()
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

def weighted_Mat(M,func):
    for n in range(1,np.shape(M)[0]):
        M[n-1,:] = func(n) * M[n-1,:]
    return M

def Limits(X):
    X_sorted=np.sort(X)
    lower_index=int(0.025*len(X))
    upper_index=int(0.975*len(X))
    return X_sorted[lower_index],X_sorted[upper_index]

def Error(lower, upper, f):
    return quad(f, lower, upper)[0]


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
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)



J=int(T/delta_time)





error=[]
trials=1

error_data=np.zeros((trials,6))
f0 = lambda n: 1
f1 = lambda n: 1/(n**2)
f2 = lambda n: 1/(n**3)
f3 = lambda n: 1/(n**7)
f4 = lambda n: 1/(n**12)
f5 = lambda n: 1/(n**20)

names=['1','1/n','$1/(n^3)$','$1/(n^7)$','$1/(n^{12})$','$1/(n^{20})$']
P=np.polynomial.polynomial.Polynomial
for _ in range(trials):
    print(_)
    X = simulation(delta_time, J, 0, model)
    lower, upper = Limits(X)

    moments_diff, moments_2_diff = Poly(X, J, degree, n_lines, P)
    M = Matrix(moments_diff, moments_2_diff)
    errors=[]
    count=0
    if _ == trials - 1:
        x = np.linspace(lower, upper)
        ax1.plot(x, Drift_(x), label='True')
    for func in [f0,f1,f2,f3,f4,f5]:
        M_= weighted_Mat(M,func)
        parameters = finding_parameters(X, J, delta_time, M_)
        f = lambda x: ((Drift_(x) - P(parameters[:-1]).__call__(x)) ** 2) * mesure(x)
        if _ == trials-1:
            ax1.plot(x, P(parameters[:-1]).__call__(x),
                     label=names[count] + ' ' + f'{"{:.4f}".format(Error(lower, upper, f))}')
        errors.append(Error(lower, upper, f))
        count += 1
    error_data[_, :] = np.array(errors)



ax1.set(xlabel='X ', ylabel='Drift(X)')
ax1.grid()
ax1.legend(loc='center right',prop={'size': 20}, bbox_to_anchor=(1, 0.5))

ax2.boxplot(error_data)

ax2.grid()
ax2.set_xticklabels([str(i) for i in ['1','1/n','$1/(n^3)$','$1/(n^7)$','$1/(n^{12})$','$1/(n^{20})$']])
ax2.set(xlabel='L', ylabel='Error')
plt.savefig(f'Weighted_poly5_(11)_(11)_Gaussian_noise.svg')


plt.show()