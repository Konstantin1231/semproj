import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
alpha_2=0
alpha_1=-1
alpha_0=0
sigma=1

'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new=X_old + alpha_1*X_old*delta_time+ np.sqrt(2*sigma*delta_time)*norm.rvs()
        sim.append(X_new)
    return sim



''' 
Estimeteurs
'''
def Moment(max_degree,X,J,delta_time):
    moments=np.ones(max_degree+1)
    V = np.array(X[0:J-1])
    for i in range(1,max_degree+1):
        moments[i]=(1/J) * np.sum(V)
        V=V*np.array(X[0:J-1])
    return moments

def Matrix_special(X,n_lignes,moments):
    M=np.zeros([n_lignes+1,3])
    M[0,:]=np.array([moments[1],moments[0],0])
    for i in range(2,n_lignes+1):
        M[i-1,:]=np.array([moments[i],moments[i-1],(i-1)*moments[i-2]])
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
    reg = LinearRegression(fit_intercept=False).fit(M,Y)
    return reg.coef_



T=2000
delta_time=0.01
J=int(T/delta_time)
T=J*delta_time
X=simulation(delta_time,J,0)

fig = plt.figure(figsize=(15, 5))
t = np.linspace(-5, +5, 100)
ax1=fig.add_subplot(231)
ax2=fig.add_subplot(232)
ax3=fig.add_subplot(233)
ax4=fig.add_subplot(234)
ax5=fig.add_subplot(235)
xx = delta_time * np.array(range(500,J,500))
ax1.plot(xx, sigma*np.ones(len(xx)))
ax2.plot(xx, alpha_2*np.ones(len(xx)))
ax3.plot(xx, alpha_1*np.ones(len(xx)))
ax4.plot(xx, alpha_0*np.ones(len(xx)))
ax5.plot(t, alpha_1 * t, label='True')
max_lines=8
for n_lines in range(4,max_lines):
    alphas_0=[]
    alphas_1=[]
    alphas_2 = []
    sigmas=[]
    for j in range(500,J,500):
        max_degree = n_lines+1
        moments = Moment(max_degree, X, j, delta_time)
        M = Matrix_special(X, n_lines, moments)
        parameters = finding_parameters(X, j, delta_time, M)
        alphas_2.append(parameters[0])
        alphas_1.append(parameters[1])
        alphas_0.append(parameters[2])
        sigmas.append(parameters[3])


    ax1.plot(xx,sigmas)
    ax2.plot(xx,alphas_2)
    ax3.plot(xx,alphas_1)
    ax4.plot(xx,alphas_0)
    ax5.plot(t, alphas_2[-1] * (t * t) + alphas_1[-1] * t + alphas_0[-1] * np.ones(len(t)), label=f'# of lines-{n_lines}')



ax1.set(xlabel='time(T) ', ylabel='Sigma')
ax1.grid()
ax2.set(xlabel='time(T) ', ylabel='Alpha_2')
ax2.grid()
ax3.set(xlabel='time(T) ', ylabel='Alpha_1')
ax3.grid()
ax4.set(xlabel='time(T) ', ylabel='Alpha_0')
ax4.grid()
ax5.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
plt.savefig(f'non-parametric_exemple({alpha_2},{alpha_1},{alpha_0},{sigma}).svg')

plt.show()

plt.show()