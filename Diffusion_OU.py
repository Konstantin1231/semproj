import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy import linalg as LA
'''
Parameters of true model
'''
alpha_1=-1
sigma_1=0
sigma_0=1
sigma_2=1
'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new = X_old + alpha_1 * X_old * delta_time + np.sqrt(2*(sigma_0 + sigma_2*X_old**2)*delta_time)*norm.rvs()
        sim.append(X_new)
    return sim

def Limits(X):
    X_sorted=np.sort(X)
    lower_index=int(0.025*len(X))
    upper_index=int(0.975*len(X))
    return X_sorted[lower_index],X_sorted[upper_index]


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



def Matrix_special(X,n_lignes,moments):

    M=np.zeros([n_lignes+1,4])
    M[0,:]=np.array([moments[1],0,0,0])
    for i in range(1,n_lignes):
        M[i,:]=np.array([moments[i+1],i*moments[i+1],i*moments[i],i*moments[i-1]])
    M[n_lignes,:]=np.array([0,moments[2],moments[1],1])
    return M

def RebuildMatrix(M):
    removed_line=M[:,0]
    new_Matrix=M[:,1:]
    return new_Matrix , removed_line

'''
Calculating coef
'''
def finding_parameters(X,J,delta_time,M):
    M , removed_line = RebuildMatrix(M)
    A1 = np.array(X[1:J]) - np.array(X[0:J - 1])
    Q= (1 /(2* J*delta_time)) * np.dot(A1, A1)
    Y=np.zeros(np.shape(M)[0])
    Y[len(Y)-1]=Q
    Y=Y-alpha_1*removed_line
    reg = LinearRegression(fit_intercept=False).fit(M,Y)
    return reg.coef_



T=10000
delta_time=0.01
J=int(T/delta_time)

X=simulation(delta_time,J,0)
lower,upper=Limits(X)
print(lower,upper)
fig = plt.figure(figsize=(15, 5))
xx=delta_time*np.array(range(10000,J,10000))
ax1=fig.add_subplot(232)
ax2=fig.add_subplot(231)
ax3=fig.add_subplot(233)
ax4=fig.add_subplot(234)
ax5=fig.add_subplot(235)
ax1.plot(xx,sigma_1*np.ones(len(xx)))
ax2.plot(xx,sigma_0*np.ones(len(xx)))
ax3.plot(xx, sigma_2*np.ones(len(xx)))
x = np.linspace(lower, upper, 100)
ax5.plot(x,sigma_2*x**2+ sigma_0 * np.ones(len(x)), label='True')
k_numbers=[]
for n_lines in [4]:

    sigmas_0=[]
    sigmas_1=[]
    sigmas_2=[]

    for j in range(10000,J,10000):

        max_degree = n_lines+3
        moments = Moment(max_degree, X, j, delta_time)
        M = Matrix_special(X, n_lines, moments)
        parameters=finding_parameters(X,j,delta_time,M)

        sigmas_0.append(parameters[2])
        sigmas_2.append(parameters[0])
        sigmas_1.append(parameters[1])
    M, removed_line = RebuildMatrix(M)
    k_numbers.append(LA.cond(np.transpose(M)@M))
    ax1.plot(xx,sigmas_1)
    ax2.plot(xx,sigmas_0)
    ax3.plot(xx, sigmas_2)

    ax5.plot(x,sigmas_2[-1]*x**2+sigmas_1[-1]*x+sigmas_0[-1]*np.ones(len(x)), label=f'# of lines-{n_lines}')

ax4.plot(k_numbers)
ax1.grid()
ax4.grid()
ax2.set(xlabel='time(T) ', ylabel='Sigma_0')
ax2.grid()
ax3.set(xlabel='time(T) ', ylabel='Sigma_2')
ax3.grid()
ax5.legend(loc='center right', bbox_to_anchor=(1.28, 0.5))
ax5.grid()
plt.savefig(f'Diffusion_OU({alpha_1},{sigma_2},{sigma_0}).svg')


plt.show()