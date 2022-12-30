import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
#condition 2alpha_0>(2sigma_0)**4
alpha_1=-1
alpha_0=1
sigma_1=0.3


cox=[alpha_1,alpha_0,sigma_1]
'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new = X_old + (alpha_1 * X_old +alpha_0)* delta_time + np.sqrt(2*(sigma_1 * X_old)*delta_time)*norm.rvs()
        sim.append(X_new)
    return sim



''' 
Estimeteurs
'''
def Moment(max_degree,X,J,delta_time,k):
    moments=np.ones(max_degree+1)
    V = np.array(X[0:J])[::k]
    for i in range(1,max_degree+1):
        moments[i]=(1/(J/k)) * np.sum(V)
        V=V*(np.array(X[0:J])[::k])
    return moments

def Matrix_special(X,n_lignes,moments):

    M=np.zeros([n_lignes+1,3])
    M[0,:]=np.array([moments[1],moments[0],0])
    for i in range(1,n_lignes):
        M[i,:]=np.array([moments[i+1],moments[i],i*(moments[i])])
    M[n_lignes,:]=np.array([0,0,moments[1]])
    return M

def Matrix_special_Lamperti(X,n_lignes,moments):
    M=np.zeros([n_lignes+1,3])
    M[0,:]=np.array([moments[0],moments[2],moments[0]])
    for i in range(1,n_lignes):
        M[i,:]=np.array([moments[i],moments[i+2],(i+1)*(moments[i])])
    M[n_lignes,:]=np.array([0,0,1])
    return M

'''
Calculating coef
'''
def finding_parameters(X,J,delta_time,M,k):

    A1 = np.array(X[1:J]) - np.array(X[0:J - 1])
    A1 = A1[::k]
    Q= k*(1 /(2* J*delta_time)) * np.dot(A1, A1)
    Y=np.zeros(np.shape(M)[0])
    Y[len(Y)-1]=Q
    print(Q)
    reg = LinearRegression(fit_intercept=False).fit(M,Y)
    return reg.coef_, Q



T=1500
delta_time=0.005
J=int(T/delta_time)
n_lines=2

k=1 #step
X=simulation(delta_time,J,alpha_0)

fig = plt.figure(figsize=(15, 5))
xx=delta_time*np.array([int(i*J/(120)) for i in range(3,120,3)])
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)



alphas1=[]
alphas0=[]
sigmas=[]

for j in [int(i*J/(120)) for i in range(3,120,3)]:
    max_degree = n_lines + 2
    moments = Moment(max_degree, X, j, delta_time,k)
    M = Matrix_special(X, n_lines, moments)
    parameters, q_c = finding_parameters(X, j, delta_time, M,k)
    sigmas.append(parameters[2])
    alphas1.append(parameters[0])
    alphas0.append(parameters[1])

ax3.plot(xx,sigma_1*np.ones(len(xx)),color='r',label='True')
ax3.plot(xx,sigmas,label='MoM')

ax2.plot(xx,alpha_0*np.ones(len(xx)),color='r',label='True')
ax2.plot(xx,alphas0,label='MoM')

ax1.plot(xx,alpha_1*np.ones(len(xx)),color='r',label='True')
ax1.plot(xx,alphas1,label='MoM')


ax1.set_ylim(bottom=-1.5, top=-0.5)
ax2.set_ylim(bottom=0.5, top=1.5)
ax3.set_ylim(bottom=0.2, top=0.4)



ax1.set(xlabel='time(T) ', ylabel='$\\alpha_{1}$')
ax1.grid()
ax2.set(xlabel='time(T) ', ylabel='$\\alpha_{0}$')
ax2.grid()
ax3.set(xlabel='time(T) ', ylabel='$\\sigma$')
ax3.grid()
ax3.legend(loc='center right', bbox_to_anchor=(1.28, 0.5))

plt.savefig(f'Cox).jpeg')


plt.show()



