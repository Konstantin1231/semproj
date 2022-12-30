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




delta_time=0.01
J=40000
T=J*delta_time
X=simulation(delta_time,J,0)

fig = plt.figure(figsize=(15, 5))
xx=delta_time*np.array(range(3000,J,1000))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
for n_lines in range(5,9):

    alphas=[]
    sigmas=[]
    for j in range(3000,J,1000):
        max_degree = n_lines
        moments = Moment(max_degree, X, j, delta_time)
        M = Matrix_special(X, n_lines, moments)
        parameters=finding_parameters(X,j,delta_time,M)
        sigmas.append(parameters[1])
        alphas.append(parameters[0])
    print(M)
    xx = delta_time * np.array(range(3000,J,1000))
    ax1.plot(xx,sigmas)
    ax2.plot(xx,alphas, label=f'# of lines-{n_lines}')




ax1.set(xlabel='time(T) ', ylabel='Sigma')
ax1.grid()
ax2.set(xlabel='time(T) ', ylabel='Alpha')
ax2.grid()
ax2.legend(loc='center right', bbox_to_anchor=(1.28, 0.5))

#plt.savefig(f'new_style_exemple({alpha},{sigma}).svg')


plt.show()