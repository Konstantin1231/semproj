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



T=10000
delta_time=0.01
J=int(T/delta_time)

X=simulation(delta_time,J,0)

fig = plt.figure(figsize=(15, 5))
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)
n_lines=2

alphas_c=[]
sigmas_c=[]
alphas=[]
sigmas=[]
error_MOM=[]
error_classic=[]
for j in range(3000,J,500):
    max_degree = n_lines
    moments = Moment(max_degree, X, j, delta_time)
    M = Matrix_special(X,n_lines, moments)
    M = Matrix_special_(n_lines, M)
    parameters=finding_parameters(X,j,delta_time,M)
    sigmas.append(parameters[1])
    alphas.append(parameters[0])
    sigmas_c.append(sigma_(X, delta_time, j)**2/2)
    alphas_c.append(alpha_(X, delta_time, j))
    error_MOM.append(np.sqrt((parameters[0]-alpha)**2+(parameters[1]-sigma)**2))
    error_classic.append(np.sqrt((alphas_c[-1]-alpha)**2+(sigmas_c[-1]-sigma)**2))

xx = delta_time * np.array(range(3000,J,500))
ax1.plot(xx,sigma*np.ones(len(xx)),color='r',label='True')
ax1.plot(xx,sigmas_c,label='classical')
ax1.plot(xx,sigmas,label='MoM')

ax2.plot(xx,alpha*np.ones(len(xx)),color='r',label='True')
ax2.plot(xx,alphas_c,label='classical')
ax2.plot(xx,alphas,label='MoM')

ax3.plot(xx,0*np.ones(len(xx)),color='r',label='True')
ax3.plot(xx,error_classic,label='classical')
ax3.plot(xx,error_MOM,label='MoM')




ax1.set(xlabel='time(T) ', ylabel='Sigma')
ax1.grid()
ax2.set(xlabel='time(T) ', ylabel='Alpha')
ax2.grid()
ax3.set(xlabel='time(T) ', ylabel='Error')
ax3.grid()
ax3.legend(loc='center right', bbox_to_anchor=(1.28, 0.5))

plt.savefig(f'results({alpha},{sigma}).svg')


plt.show()