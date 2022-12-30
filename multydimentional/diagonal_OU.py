import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
#condition 2alpha_0>(2sigma_0)**4
alpha_1=2
alpha_2=1
alpha_3=1
alpha_4=1
A=np.array([[alpha_1,alpha_2],[alpha_3,alpha_4]])
sigma_1 = 1/2
sigma_2 = 1/2
X_0=np.array([0,0])
exact = [-alpha_1,-alpha_2,-alpha_4,sigma_1,sigma_2]

'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=np.zeros((2,J+1))
    sim[:,0]=X_0
    for i in range(1,J+1):
        X_old  = sim[:,i-1]
        X_new_2 =  X_old[1] - (alpha_3*X_old[0]+alpha_4*X_old[1]) * delta_time + np.sqrt(2*sigma_1 * delta_time)*norm.rvs()
        X_new_1 =  X_old[0] - (alpha_1*X_old[0]+alpha_2*X_old[1]) * delta_time + np.sqrt(2*sigma_1 * delta_time)*norm.rvs()

        sim[:,i]=np.array([X_new_1,X_new_2])
    return sim



''' 
Estimeteurs
'''


'''
Calculating coef
'''
def finding_parameters(X1,X2,J,delta_time,M):
    X_1_ = X1
    X_2_ = X2

    A2 = np.array(X_2_[1:J]) - np.array(X_2_[0:J - 1])
    Q2= (1 /(2* J*delta_time)) * np.dot(A2, A2)
    A1 = np.array(X_1_[1:J]) - np.array(X_1_[0:J - 1])
    Q1 = (1 / (2 * J * delta_time)) * np.dot(A1, A1)
    Y=np.zeros(np.shape(M)[0])
    Y[len(Y)-2]=Q1
    Y[len(Y)-1]=Q2
    print(Q1,'     ',Q2)
    reg = LinearRegression(fit_intercept=False).fit(M,Y)
    return reg.coef_


k=1
T=1500
delta_time=0.01
J=int(T/delta_time)
X=np.array(simulation(delta_time,J,X_0))

















fig = plt.figure(figsize=(30, 12))
plt.rcParams['font.size'] = '15'
ax1=fig.add_subplot(231)
ax2=fig.add_subplot(232)
ax3=fig.add_subplot(233)
ax4=fig.add_subplot(234)
ax5=fig.add_subplot(235)
ax6=fig.add_subplot(236)

xx=delta_time*np.array([int(i*J/(120)) for i in range(3,120,3)])

ax1.plot([0,T],-alpha_1*np.array([1,1]),color='red')
ax2.plot([0,T],-alpha_2*np.array([1,1]),color='red')
ax3.plot([0,T],-alpha_4*np.array([1,1]),color='red')
ax4.plot([0,T],sigma_1*np.array([1,1]),color='red')
ax5.plot([0,T],sigma_2*np.array([1,1]),color='red')

alphas_1=[]
alphas_2=[]
alphas_3=[]
sigmas_1=[]
sigmas_2=[]
error=[]
X1=X[0,:]
X2=X[1,:]

for j in [int(i*J/(120)) for i in range(3,120,3)]:
    X_1=X1[0:j]
    X_2=X2[0:j]

    M_L = [[np.mean(X_1 ** 2), np.mean(X_1 * X_2), 0, 1, 0],
           [0, np.mean(X_1 * X_2), np.mean(X_2 ** 2), 0, 1],
           [np.mean(X_1 * X_2), np.mean(X_2 ** 2) + np.mean(X_1 ** 2), np.mean(X_1 * X_2), 0, 0],
           [0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1]
           ]

    parameters = finding_parameters(X_1,X_2, j, delta_time, M_L)

    alphas_1.append(parameters[0])
    alphas_2.append(parameters[1])
    alphas_3.append(parameters[2])
    sigmas_1.append(parameters[3])
    sigmas_2.append(parameters[4])
    error.append(np.sqrt(np.sum((np.array(exact)-parameters)**2)))


ax1.plot(xx,alphas_1)
ax2.plot(xx,alphas_2)
ax3.plot(xx,alphas_3)
ax4.plot(xx,sigmas_1)
ax5.plot(xx,sigmas_2)
ax6.plot(xx,error)
ax1.set(xlabel='time(T) ', ylabel='$\\alpha_{1}$')
ax1.grid()
ax2.set(xlabel='time(T) ', ylabel='$\\alpha_{2}$')
ax2.grid()
ax3.set(xlabel='time(T) ', ylabel='$\\alpha_{3}$')
ax3.grid()
ax4.set(xlabel='time(T) ', ylabel='$\\sigma_{1}$')
ax4.grid()
ax5.set(xlabel='time(T) ', ylabel='$\\sigma_{2}$')
ax5.grid()
ax6.set(xlabel='time(T) ', ylabel='Error')
ax6.grid()

ax1.set_ylim(bottom=-2.2, top=-1.8)
ax2.set_ylim(bottom=-0.7, top=-1.3)
ax3.set_ylim(bottom=-0.7, top=-1.3)
ax4.set_ylim(bottom=0.3, top=0.7)
ax5.set_ylim(bottom=0.3, top=0.7)


plt.savefig(f'Multidimentional_trial1.svg')
plt.show()



