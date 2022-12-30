import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
alpha_1=1
alpha_3=-1
sigma=1

'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new=X_old +(alpha_3*(X_old**3)+ alpha_1*X_old)*delta_time+ np.sqrt(2*sigma*delta_time)*norm.rvs()
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
    reg = LinearRegression(fit_intercept=False).fit(M,Y)
    return reg.coef_



T=5000
delta_time=0.01
J=int(T/delta_time)



fig = plt.figure(figsize=(15, 10))

ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
ax1.plot([0,T],[0,0], color='red')
ax2.plot([0,T],[0,0], color='red',label=' ')

xx = delta_time * np.array(range(10000, J, 5000))
xx_=delta_time*30000+delta_time*np.array(range(10000, J-30000, 5000))
n_lines=4
max_degree = 6

for X_0 in [0,3,5,7,10]:
    X = simulation(delta_time, J, X_0)
    X_= X[30000:]
    error=[]
    error_removed=[]
    for j in range(10000, J, 5000):
        moments = Moment(max_degree, X, j, delta_time)
        M = Matrix_special(X, n_lines, moments)
        parameters = finding_parameters(X, j, delta_time, M)
        error.append(np.sqrt( np.sum((parameters-[-1,1,1])**2 )))
    for j in range(10000, J-30000, 5000):
        moments = Moment(max_degree, X_, j, delta_time)
        M = Matrix_special(X_, n_lines, moments)
        parameters = finding_parameters(X_, j, delta_time, M)
        error_removed.append(np.sqrt(np.sum((parameters - [-1, 1, 1]) ** 2)))

    ax1.plot(xx,error)
    ax2.plot(xx_,error_removed, label=f'X(0)={X_0}')



ax1.set_ylim(bottom=0, top=1.6)
ax2.set_ylim(bottom=0, top=1.6)
ax1.set(xlabel='time(T) ', ylabel='Error')
ax1.grid()
ax2.set(xlabel='time(T) ', ylabel='Error')
ax2.grid()
ax2.legend(loc='center right', bbox_to_anchor=(1.28, 0.5))
plt.savefig(f'different_starting_point(1).jpeg')

plt.show()