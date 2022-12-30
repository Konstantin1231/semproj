import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
#condition 2alpha_0>(2sigma_0)**4
alpha_3=-2
alpha_1=-1
sigma_0=1
sigma_2=1/2
k=1

'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new = X_old + alpha_1 * ( X_old**3 + X_old)* delta_time + np.sqrt(2*(sigma_0)*delta_time)*norm.rvs()
        sim.append(X_new)
    return sim

def simulation2(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new = X_old + alpha_1 * ( X_old**3 + X_old) * delta_time + np.sqrt(2*((sigma_0 +sigma_2 * X_old**2)**2)*delta_time)*norm.rvs()
        sim.append(X_new)
        if(np.abs(X_new)>1000):
            print('hi')
    return sim


''' 
Estimeteurs
'''

'''
Calculating coef
'''
def finding_parameters(X,J,delta_time,k):
    A1 = np.array(X[1:J]) - np.array(X[0:J - 1])
    A1 = A1[::k]
    Q= k*(1 /(2* J*delta_time)) * np.dot(A1, A1)
    return Q

def Moment_variance(max_degree,X,J,delta_time,k):
    moments=np.ones(max_degree+1)
    V = np.array(X[0:J])[::k]
    for i in range(1,max_degree+1):
        moments[i]=np.mean(V)
        V=V*(np.array(X[0:J])[::k])
    Q = sigma_0**2 + 2*sigma_0*sigma_2*moments[2]+ sigma_2**2 * moments[4]

    return Q


k=100
T=400
T_=250
delta_time=0.001
X_0=2
J=int(T/delta_time)
J_=int(T_/delta_time)
X=simulation(delta_time,J,X_0)
j=J
Q_c=[]
Q_L=[]
Q=[]
Error=[]
Error_=[]
Error__=[]
for trial in range(1):
    X_L = simulation2(delta_time, J, X_0)
    for j in [int(i * J / (120)) for i in range(9, 120, 3)]:
        q_c = finding_parameters(X_L,j,delta_time,k)
        q = Moment_variance(4,X_L,j,delta_time,k)
        Error__.append(np.abs(q_c-q)/q_c)

        q_c = finding_parameters(X,j,delta_time,k)
        Error_.append(np.abs(q_c-sigma_0)/sigma_0)
        print(q_c)


J_=8
fig = plt.figure(figsize=(9, 4))
ax1=fig.add_subplot(111)
xx=delta_time*np.array([int(i*J/(120)) for i in range(9,120,3)])
ax1.plot(xx[J_:],Error__[J_:])
#ax1.boxplot([Error_,Error__])
#ax1.set_xticklabels(['Additive','Multiplicative'])
ax1.set( ylabel='$Relative error$')
ax1.grid()
plt.show()
plt.savefig(f'NewQ_error_between_lines.jpeg')


"""

trials=20

Q_c=[]
Q_L=[]

T=600

J=int(T/delta_time)
j=J
for trial in range(trials):
    X = simulation(delta_time, J, X_0)

    X_L = simulation2(delta_time, J, X_0)


    q_c = finding_parameters(X, j, delta_time,k)
    q_L =finding_parameters(X_L,j, delta_time,k)
    Q_c.append(q_c)
    Q_L.append(q_L)





ax2=fig.add_subplot(122)
ax2.boxplot([Q_c,Q_L])
ax2.set_xticklabels(['Additive','Multiplicative'])
ax2.set( ylabel='$Q_{T}/2T$')
ax2.grid()
plt.savefig(f'2compare_quadratique_estimateur_stat({alpha_1},{alpha_3},{sigma_0},{sigma_2}Time={T}).svg')
plt.show()
"""