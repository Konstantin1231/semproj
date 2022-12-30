import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
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
def sigma_(X,delta_time,J):
    A1=np.array(X[1:J]) - np.array(X[0:J-1])
    return np.sqrt((1/(J*delta_time)) * np.dot(A1,A1))

def alpha_(X,delta_time,J):
    V=np.array(X[0:J - 1])
    A1 =np.dot(np.array(X[1:J]) - V,V)
    A2= delta_time * np.dot(V,V)
    return A1/A2

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

def Matrix_special_(n_lignes,M):
    if n_lignes%2==0:
        size=n_lignes/2
    else:
        size=(n_lignes-1)/2
    M_=np.zeros([int(size)+1,2])
    for i in range(0,int(size)):
            M_[i,:]=M[1+2*i,:]
    M_[int(size), :] = np.array([0, 1])
    return M_

def weighted_Mat(M,func):
    for n in range(1,np.shape(M)[0]):
        M[n-1,:] = func(n) * M[n-1,:]
    return M



'''
Calculating coef
'''
def finding_parameters(X,J,delta_time,M):
    A1 = np.array(X[1:J]) - np.array(X[0:J - 1])
    Q= (1 /(2* J*delta_time)) * np.dot(A1, A1)
    Y=np.zeros(np.shape(M)[0])
    Y[len(Y)-1]=Q
    print(Q)
    reg = LinearRegression(fit_intercept=False).fit(M,Y)
    return reg.coef_



T=50000
delta_time=0.01
J=int(T/delta_time)

X=simulation(delta_time,J,0)

fig = plt.figure(figsize=(15, 10))
ax1=fig.add_subplot(111)
n_lines=2


alphas=[]
sigmas=[]
error_MOM=[]
for j in range(20000,J,20000):
    max_degree = n_lines
    moments = Moment(max_degree, X[j:j+20000+1], 20000, delta_time)
    M = Matrix_special(X,n_lines, moments)
    M = Matrix_special_(n_lines, M)
    parameters=finding_parameters(X[j:j+20000],20000,delta_time,M)
    sigmas.append(parameters[1])
    alphas.append(parameters[0])


plt.hist(alphas,bins=9)
plt.savefig('5Alpha_distribution.jpeg')
plt.show()
