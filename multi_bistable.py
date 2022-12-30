import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
alpha_3=-1
alpha_2=0
alpha_1=1
alpha_0=0
sigma_0=0.4
sigma_2=0.1
'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new = X_old +(alpha_3*(X_old**3)+ alpha_1*X_old)*delta_time + np.sqrt(2*(sigma_0 + sigma_2*X_old**2)*delta_time)*norm.rvs()
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

    M=np.zeros([n_lignes+1,4])
    M[0,:]=np.array([moments[3],moments[1],0,0])
    for i in range(1,n_lignes):
        M[i,:]=np.array([moments[i+3],moments[i+1],i*moments[i+1],i*moments[i-1]])
    M[n_lignes,:]=np.array([0,0,moments[2],1])
    return M

def Matrix_special_(X,n_lignes,moments):
    M=np.zeros([n_lignes+1,3])
    M[0,:]=np.array([moments[3],moments[1],0])
    for i in range(2,n_lignes+1):
        M[i-1,:]=np.array([moments[i+2],moments[i],(i-1)*moments[i-2]])
    M[n_lignes,:]=np.array([0,0,1])
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



T=15000
delta_time=0.01
J=int(T/delta_time)

X=simulation(delta_time,J,0)


fig = plt.figure(figsize=(15, 10))
xx=delta_time*np.array(range(1000,J,2000))
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
ax4=fig.add_subplot(224)
ax1.plot(xx,alpha_3*np.ones(len(xx)))
ax2.plot(xx,alpha_1*np.ones(len(xx)))
ax3.plot(xx,sigma_0*np.ones(len(xx)))
ax4.plot(xx, sigma_2*np.ones(len(xx)), label='True')
for n_lines in range(6,8):
    alphas_3=[]
    alphas_1=[]
    sigmas_0=[]
    sigmas_2=[]

    for j in range(1000,J,2000):

        max_degree = n_lines+3
        moments = Moment(max_degree, X, j, delta_time)
        M = Matrix_special(X, n_lines, moments)
        parameters=finding_parameters(X,j,delta_time,M)

        sigmas_0.append(parameters[3])
        sigmas_2.append(parameters[2])
        alphas_1.append(parameters[1])
        alphas_3.append(parameters[0])

    ax1.plot(xx,alphas_3)
    ax2.plot(xx, alphas_1)
    ax3.plot(xx,sigmas_0)
    ax4.plot(xx, sigmas_2, label=f'# of lines-{n_lines}')



ax1.set(xlabel='time(T) ', ylabel='Alpha_3')
ax1.grid()
ax2.set(xlabel='time(T) ', ylabel='Alpha_1')
ax2.grid()
ax3.set(xlabel='time(T) ', ylabel='Sigma_0')
ax3.grid()
ax4.set(xlabel='time(T) ', ylabel='Sigma_2')
ax4.grid()
ax4.legend(loc='center right', bbox_to_anchor=(1.28, 0.5))

plt.savefig(f'Multiplicative_bistable({alpha_3},{alpha_1},{sigma_2},{sigma_0}).svg')


plt.show()