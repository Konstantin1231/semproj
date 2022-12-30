import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy import linalg as LA
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
    V = np.array(X[0:J])
    for i in range(1,max_degree+1):
        moments[i]=(1/J) * np.sum(V)
        V=V*np.array(X[0:J])
    return moments

def Matrix_special(X,n_lignes,moments):
    M=np.zeros([n_lignes+1,3])
    M[0,:]=np.array([moments[3],moments[1],0])
    for i in range(2,n_lignes+1):
        M[i-1,:]=np.array([moments[i+2],moments[i],(i-1)*moments[i-2]])
    M[n_lignes,:]=np.array([0,0,1])
    return M

def Matrix_special_(n_lignes,M):
    M_=np.array(M)
    for i in range(np.shape(M)[0]):
        if i>=2 and i%2==0:
            M_[i,:]=np.zeros(np.shape(M)[1])
    M_[np.shape(M)[0]-1, :] = np.array([0,0, 1])
    return M_



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



T=1000
delta_time=0.01
J=int(T/delta_time)



fig = plt.figure(figsize=(15, 5))

ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
error_data=[]
kond_number_data=[]
error_data_removed=[]
kond_number_data_removed=[]
for n_lines in range(4,10,2):
    errors=[]
    error_removed=[]
    kond_number=[]
    kond_number_removed=[]
    for _ in range(30):
        X = simulation(delta_time, J, 0)
        j=J
        max_degree = n_lines+2
        moments = Moment(max_degree, X, j, delta_time)
        M = Matrix_special(X, n_lines, moments)
        M_= Matrix_special_(n_lines,M)
        kond_number_removed.append(LA.cond(np.transpose(M_)@M_))
        k=LA.cond(np.transpose(M)@M)
        kond_number.append(k)
        parameters=finding_parameters(X,j,delta_time,M)
        parameters_=finding_parameters(X,j,delta_time,M_)
        errors.append(np.sqrt((parameters[0]-alpha_3)**2+(parameters[2]-sigma)**2+(parameters[1]-alpha_1)**2))
        error_removed.append(np.sqrt((parameters_[0]-alpha_3)**2+(parameters_[2]-sigma)**2+(parameters_[1]-alpha_1)**2))

    print(n_lines, parameters, errors)
    kond_number_data.append(np.mean(kond_number))
    kond_number_data_removed.append(np.mean(kond_number_removed))
    error_data.append(errors)
    error_data_removed.append(error_removed)
ax1.boxplot(error_data)
ax2.boxplot(error_data_removed)
ax1.grid()
ax2.grid()
ax1.set_ylim(bottom=0, top=2)
ax2.set_ylim(bottom=0, top=2)
ax1.set_xticklabels([str(i) for i in range(4, 10, 2)])
ax1.set(xlabel='# of equations ', ylabel='Error')
ax2.set_xticklabels([str(i) for i in range(4, 10, 2)])
ax2.set(xlabel='# of equations ', ylabel='Error')
plt.savefig(f'error_removed({alpha_3},{alpha_1},{sigma}).svg')

plt.show()
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(111)

ax1.scatter(np.array(range(4, 10, 2)), kond_number_data, label='Full matrix')
ax1.scatter(np.array(range(4, 10, 2)), kond_number_data_removed, label='No zero lines')
ax1.grid()
ax1.set(xlabel='# of equations ', ylabel='Conditional number K(M)')

plt.savefig(f'Cond_number_error_removed({alpha_3},{alpha_1},{sigma}).svg')
plt.show()

