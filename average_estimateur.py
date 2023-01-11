
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
'''
Parameters of true model
'''
alpha_1=1
alpha_3=-1
sigma=1

bi=np.array([alpha_3,alpha_1,sigma])

alpha_1_=-3
alpha_0_=4
sigma_1_=1

cox=np.array([alpha_1_,alpha_0_,sigma_1_])
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

def simulation_(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new = X_old + (alpha_1_ * X_old +alpha_0_)* delta_time + np.sqrt(2*(sigma_1_ * np.abs(X_old))*delta_time)*norm.rvs()
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

def Matrix_special_(X,n_lignes,moments):

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
    reg = LinearRegression(fit_intercept=False).fit(M,Y)
    return reg.coef_



T=1200
delta_time=0.01
J=int(T/delta_time)
n_lines=4
j=J
trials=25
K=[1,2,3,4,5,6]
total_error_average = np.zeros((trials,len(K)+1))
total_error_moments = np.zeros((trials,len(K)+1))
max_degree = n_lines + 3
for trial in range(trials):
    X = simulation(delta_time, J, 0)
    alphas_3 = []
    alphas_1 = []
    sigmas_0 = []
    X_total=[]
    X_total.append(X)
    moments = Moment(max_degree, X, J, delta_time)
    M = Matrix_special(X, n_lines, moments)
    parameters = finding_parameters(X, J, delta_time, M)
    total_error_average[trial,0]=np.sqrt(np.sum((parameters-bi)**2))
    total_error_moments[trial,0]=total_error_average[trial,0]

    for i in K:

        for j in range(2**(i-1)):
            X_new = simulation(delta_time, J, 0)
            X_total=np.append(X_total,X_new)
            print(np.shape(np.array(X_total)))
            moments = Moment(max_degree, X_new, J, delta_time)
            M = Matrix_special(X_new, n_lines, moments)
            parameters = finding_parameters(X_new, J, delta_time, M)

            sigmas_0.append(parameters[2])
            alphas_1.append(parameters[1])
            alphas_3.append(parameters[0])

        parameters_average=np.array([np.mean(alphas_3),np.mean(alphas_1),np.mean(sigmas_0)])

        moments = Moment(max_degree, X_total, len(X_total), delta_time)
        M = Matrix_special(X_total, n_lines, moments)
        parameters_total = finding_parameters(X_total, len(X_total), delta_time, M)

        total_error_average[trial, i] = np.sqrt(np.sum((parameters_average - bi) ** 2))
        total_error_moments[trial, i] = np.sqrt(np.sum((parameters_total - bi) ** 2))
        print(total_error_moments)



fig = plt.figure(figsize=(20, 5))

ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)

ax1.boxplot(total_error_moments)
ax2.boxplot(total_error_average)



ax1.set( ylabel='Error')
ax1.set_xticklabels(['1']+[str(2**i) for i in K])
ax1.grid()
ax2.set(ylabel='Error')
ax2.set_xticklabels(['1']+[str(2**i) for i in K])
ax2.grid()
ax1.set_ylim(bottom=0, top=0.6)
ax2.set_ylim(bottom=0, top=0.6)

plt.savefig(f'Averaging_estimateurs_Vs_averaging_estimateurs_{T}.svg')


plt.show()
"""
not working 
"""