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
alpha_7=-0
alpha_6=0
alpha_5=0
alpha_4=0
alpha_3=-0.6
alpha_2=0
alpha_1=0.2
alpha_0=0
sigma_0=0.3
sigma_1=0
sigma_2=0
true=[alpha_0,alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,sigma_0,sigma_1,sigma_2]

"""STANDARD"""
'''
#Diffusion=np.polynomial.polynomial.Polynomial([sigma_0,sigma_1,sigma_2])
Drift=np.polynomial.polynomial.Polynomial([alpha_0,alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6,alpha_7])

Drift_= lambda x: (np.polynomial.polynomial.Polynomial([alpha_0,alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6,alpha_7]).__call__(x))
#Drift_=lambda x: np.polynomial.polynomial.Polynomial([alpha_0,alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6,alpha_7])

V= lambda x: Drift.integ(1).__call__(x)
inv_mesure=lambda x: np.exp(V(x)/sigma_0)
mesure = lambda x: inv_mesure(x)/quad(inv_mesure,-np.inf,np.inf)[0]

'''
"""SPECIAL"""

#Diffusion=np.polynomial.polynomial.Polynomial([sigma_0,sigma_1,sigma_2])
Drift= lambda x: np.sin(2*x) - x

#+ 1 - 0.1*x
Drift_= lambda x: np.sin(2*x) - x
#Drift_=lambda x: np.polynomial.polynomial.Polynomial([alpha_0,alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6,alpha_7])

#x - 0.1*(x**2)/2
V= lambda x: -np.cos(2*x)/2 - (x**2)/2
inv_mesure=lambda x: np.exp(V(x)/sigma_0)
mesure = lambda x: inv_mesure(x)/quad(inv_mesure,-np.inf,np.inf)[0]






"'SETTING"""
T=1000
delta_time=0.01
#np.polynomial.chebyshev.Chebyshev
#np.polynomial.polynomial.Polynomial
#np.polynomial.hermite.Hermite
#numpy.polynomial.legendre.Legendre
#np.polynomial.laguerre.Laguerre
degree=7
n_lines=7
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
        """standard"""
        '''
        sim=[]
        sim.append(X_0)
        for _ in range(J):
            X_old=sim[len(sim)-1]
            X_new=X_old + Drift_.__call__(X_old)*delta_time+ np.sqrt(2*sigma_0*delta_time)*norm.rvs()
            sim.append(X_new)
        '''
        """special"""
        sim=[]
        sim.append(X_0)
        for _ in range(J):
            X_old=sim[len(sim)-1]
            X_new=X_old + Drift(X_old)*delta_time+ np.sqrt(2*sigma_0*delta_time)*norm.rvs()
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
ax1=fig.add_subplot(111)



J=int(T/delta_time)
X = simulation(delta_time, J, 0,model)
lower,upper=Limits(X)
x=np.linspace(lower,upper,100)
ax1.plot(x,Drift_(x),label='True')
count=0
error=[]
#[np.polynomial.polynomial.Polynomial, np.polynomial.chebyshev.Chebyshev, np.polynomial.hermite.Hermite,np.polynomial.legendre.Legendre,np.polynomial.laguerre.Laguerre]
for P in [np.polynomial.polynomial.Polynomial, np.polynomial.chebyshev.Chebyshev, np.polynomial.hermite.Hermite,np.polynomial.legendre.Legendre,np.polynomial.laguerre.Laguerre]:
    moments_diff, moments_2_diff=Poly(X,J,degree,n_lines,P)
    M = Matrix(moments_diff,moments_2_diff)
    parameters=finding_parameters(X,J,delta_time,M)
    parameters_int=100*M
    parameters_int=parameters_int.astype(int)
    print(parameters)
    print('')
    print('')
    print('')
    f = lambda x: ((Drift_(x) - P(parameters[:-1]).__call__(x)) ** 2) * mesure(x)
    ax1.plot(x, P(parameters[:-1]).__call__(x), label=names[count] + ' ' + f'{"{:.4f}".format(Error(lower,upper,f))}')
    count+=1


ax1.set(xlabel='X ', ylabel='Drift(X)')
ax1.grid()
ax1.legend(loc='center right',prop={'size': 20}, bbox_to_anchor=(1, 0.5))

#plt.savefig(f'polynomials({alpha_5},{alpha_4},{alpha_3},{alpha_1},{alpha_0},{sigma_0}).jpeg')
plt.savefig(f'special_func_{degree}_degree.jpeg')

plt.show()





