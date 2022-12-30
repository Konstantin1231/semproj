import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
alpha_1=-5
sigma_0=1
sigma_2=0.5
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
    M[0,:]=np.array([moments[1],0,0])
    for i in range(1,n_lignes):
        M[i,:]=np.array([moments[i+1],i*moments[i+1],i*moments[i-1]])
    M[n_lignes,:]=np.array([0,moments[2],1])
    return M

def Zeros(n_lignes,M):
    M_=np.array(M)
    for i in range(np.shape(M)[0]):
        if  i%2==0:
            M_[i,:]=np.zeros(np.shape(M)[1])
    M_[np.shape(M)[0]-1, :] = np.array([0,moments[2],1])
    return M_

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



T=6000
delta_time=0.001
J=int(T/delta_time)

X=simulation(delta_time,J,0)


fig = plt.figure(figsize=(20, 7))
xx=delta_time*np.array([int(i*J/(30)) for i in range(1,30,3)])
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)
ax1.plot(xx,alpha_1*np.ones(len(xx)))
ax2.plot(xx,sigma_0*np.ones(len(xx)))
ax3.plot(xx, sigma_2*np.ones(len(xx)), label='True')
for n_lines in range(4,5):


    lphas_1=[]
    igmas_0=[]
    igmas_2=[]

    for j in [int(i*J/(30)) for i in range(1,30,3)]:

        max_degree = n_lines+3
        moments = Moment(max_degree, X, j, delta_time)
        M = Matrix_special(X, n_lines, moments)
        m = Zeros(n_lines,M)
        parameters=finding_parameters(X,j,delta_time,M)
        parameter = finding_parameters(X, j, delta_time, m)
        igmas_0.append(parameter[2])
        igmas_2.append(parameter[1])
        lphas_1.append(parameter[0])


    ax1.plot(xx, lphas_1)
    ax2.plot(xx, igmas_0)
    #ax3.plot(xx, igmas_2, label=f'# of lines-{n_lines}')
    ax3.plot(xx, igmas_2, label=f'MoM')

ax1.set(xlabel='time(T) ', ylabel='alpha _1')
ax1.grid()
ax2.set(xlabel='time(T) ', ylabel='Sigma_0')
ax2.grid()
ax3.set(xlabel='time(T) ', ylabel='Sigma_2')
ax3.grid()
ax3.legend(loc='center right',prop={'size': 10}, bbox_to_anchor=(1.17, 0.5))
ax1.set_ylim(bottom=alpha_1-1, top=alpha_1+1)
ax2.set_ylim(bottom=sigma_0-0.5, top=sigma_0+0.5)
ax3.set_ylim(bottom=sigma_2-0.5, top=sigma_2+0.5)
plt.savefig(f'Multiplicative_OU({alpha_1},{sigma_2},{sigma_0}).svg')


plt.show()
