import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
alpha_1=-1
alpha_0=1
sigma_1=1

'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new = X_old + (alpha_1 * X_old +alpha_0)* delta_time + np.sqrt(2*(sigma_1 * np.abs(X_old))*delta_time)*norm.rvs()
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

def Matrix_special(X,n_lignes,moments):

    M=np.zeros([n_lignes+1,3])
    M[0,:]=np.array([moments[1],moments[0],0])
    for i in range(1,n_lignes):
        M[i,:]=np.array([moments[i+1],moments[i],i*(moments[i])])
    M[n_lignes,:]=np.array([0,0,moments[1]])
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
n_lines=4
X=simulation(delta_time,J,2)
alphas_1 = []
alphas_0 = []
sigmas_1 = []

for j in [int(J/10),int(J/9),int(J/8),int(J/7),int(J/6),int(J/4),int(J/3),int(J/2),int(J/1)]:
    max_degree = n_lines + 3
    moments = Moment(max_degree, X, j, delta_time)
    M = Matrix_special(X, n_lines, moments)
    parameters = finding_parameters(X, j, delta_time, M)

    sigmas_1.append(parameters[2])
    alphas_0.append(parameters[1])
    alphas_1.append(parameters[0])




fig = plt.figure(figsize=(15, 5))

ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)
xx=delta_time*np.array([int(J/10),int(J/9),int(J/8),int(J/7),int(J/6),int(J/4),int(J/3),int(J/2),int(J/1)])
ax1.plot(xx,alphas_0)
ax2.plot(xx,alphas_1)
ax3.plot(xx, sigmas_1)


ax1.set(xlabel='time(T) ', ylabel='Alpha_0')
ax1.grid()
ax2.set(xlabel='time(T) ', ylabel='Alpha_1')
ax2.grid()
ax3.set(xlabel='time(T) ', ylabel='Sigma_0')
ax3.grid()

#plt.savefig(f'Lamperti({alpha_1},{sigma_0}).svg')


plt.show()