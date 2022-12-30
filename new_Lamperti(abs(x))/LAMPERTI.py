import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
alpha_1=-2
alpha_0=5
sigma_1=3

lamperti=[(2*alpha_0-sigma_1),alpha_1/2,sigma_1]
cox=[alpha_1,alpha_0,sigma_1]
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
def finding_parameters(X,J,delta_time,M):

    A1 = np.array(X[1:J]) - np.array(X[0:J - 1])
    Q= (1 /(2* J*delta_time)) * np.dot(A1, A1)
    Y=np.zeros(np.shape(M)[0])
    Y[len(Y)-1]=Q
    print(Q)
    reg = LinearRegression(fit_intercept=False).fit(M,Y)
    return reg.coef_, Q



T=1000
delta_time=0.01
J=int(T/delta_time)
n_lines=4
n_lines_L=4
X=simulation(delta_time,J,alpha_0)
X_L=2*np.sqrt(np.abs(np.array(X)))
error=[]
error_L=[]
Q_c=[]
Q_L=[]
plt.plot(X)
plt.show()
for j in [int(i*J/(30)) for i in range(3,30,3)]:
    max_degree = n_lines + 3
    moments = Moment(max_degree, X, j, delta_time)
    moments_L= Moment(max_degree, X_L, j, delta_time)

    M = Matrix_special(X, n_lines, moments)
    M_L = Matrix_special_Lamperti(X_L, n_lines, moments_L)

    parameters, q_c = finding_parameters(X, j, delta_time, M)
    parameters_L, q_L=finding_parameters(X_L,j, delta_time, M_L)
    error.append(np.sqrt(np.sum((parameters-cox)**2))/(np.sqrt(np.sum(np.array(cox)**2))))
    error_L.append(np.sqrt(np.sum((parameters_L-lamperti)**2))/(np.sqrt(np.sum(np.array(lamperti)**2))))
    Q_c.append(q_c)
    Q_L.append(q_L)
    print(parameters_L,'   ',parameters)



fig = plt.figure(figsize=(15, 5))
ax1=fig.add_subplot(111)
xx=delta_time*np.array([int(i*J/(30)) for i in range(3,30,3)])
ax1.plot(xx,error)
ax1.plot(xx,error_L)

ax1.set(xlabel='time(T) ', ylabel='Relative Error')
ax1.grid()
plt.savefig(f'6_lines_MAIN_delta_time_{delta_time}_Lamperti_over_time({alpha_1},{alpha_0},{sigma_1},Time={T}).svg')
plt.show()


fig = plt.figure(figsize=(15, 5))
ax1=fig.add_subplot(111)
xx=delta_time*np.array([int(i*J/(30)) for i in range(3,30,3)])
ax1.plot(xx,Q_c)
ax1.plot(xx,Q_L)
ax1.set(xlabel='time(T) ', ylabel='$Q_{T}/2T$')
ax1.grid()
plt.savefig(f'6_lines_MAIN_delta_time_{delta_time}_compare_quadratique_estimateur_over_time({alpha_1},{alpha_0},{sigma_1},Time={T}).svg')
plt.show()





fig = plt.figure(figsize=(15, 10))
ax1=fig.add_subplot(111)
trials=20
error_data=[]
error_data_L=[]
Q_c=[]
Q_L=[]
j=J
for trial in range(trials):
    X = simulation(delta_time, J, alpha_0)
    X_L = 2 * np.sqrt(np.abs(np.array(X)))
    max_degree = n_lines + 3
    moments = Moment(max_degree, X, j, delta_time)
    moments_L= Moment(max_degree, X_L, j, delta_time)

    M = Matrix_special(X, n_lines, moments)
    M_L = Matrix_special_Lamperti(X_L, n_lines, moments_L)

    parameters, q_c = finding_parameters(X, j, delta_time, M)
    parameters_L, q_L =finding_parameters(X_L,j, delta_time, M_L)
    Q_c.append(q_c)
    Q_L.append(q_L)
    error_data.append(np.sqrt(np.sum((parameters-cox)**2))/(np.sqrt(np.sum(np.array(cox)**2))))
    error_data_L.append(np.sqrt(np.sum((parameters_L-lamperti)**2))/(np.sqrt(np.sum(np.array(lamperti)**2))))



ax1.boxplot([error_data,error_data_L])
ax1.set_xticklabels(['standard','Lamperti'])
ax1.set( ylabel='Relative Error')
ax1.grid()
plt.savefig(f'6_lines_MAIN_small_delta_time_{delta_time}_Lamperti_error({alpha_1},{alpha_0},{sigma_1},Time={T}).svg')
plt.show()

fig = plt.figure(figsize=(15, 5))
ax1=fig.add_subplot(111)
ax1.boxplot([Q_c,Q_L])
ax1.set_xticklabels(['standard','Lamperti'])
ax1.set( ylabel='$Q_{T}/2T$')
ax1.grid()
plt.savefig(f'6_lines_MAIN_delta_time_{delta_time}__compare_quadratique_estimateur_stat({alpha_1},{alpha_0},{sigma_1},Time={T}).svg')
plt.show()
