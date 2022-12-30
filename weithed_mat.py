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

def Matrix_special_(n_lignes,M):
    M_=np.array(M)
    for i in range(np.shape(M)[0]):
        if i>=2 and i%2==0:
            M_[i,:]=np.zeros(np.shape(M)[1])
    M_[np.shape(M)[0]-1, :] = np.array([0,0, 1])
    return M_
def weighted_Mat(M,func):
    for n in range(1,np.shape(M)[0]):
        M[n-1,:] = func(n) * M[n-1,:]
    return M

T=5000
delta_time=0.01
J=int(T/delta_time)



fig = plt.figure(figsize=(15, 5))
trials=30
ax1=fig.add_subplot(111)
error_data=np.zeros((trials,3))
n_lines=14
f0= lambda n: 1
f1= lambda n: 1/(2*n)
f2= lambda n: 1/(n**2)
f3 = lambda n: 1/np.log(n+1)
f4 = lambda n: 1/(n*np.log(n+1))
max_degree = n_lines+2
for _ in range(trials):
    X = simulation(delta_time, J, 0)
    j = J
    moments = Moment(max_degree, X, j, delta_time)
    M = Matrix_special(X, n_lines, moments)
    M = Matrix_special_(n_lines,M)
    errors = []
    kond_number = []
    for func in [f0,f1,f2]:
        M_=weighted_Mat(M,func)
        parameters = finding_parameters(X, j, delta_time, M_)
        errors.append(np.sqrt((parameters[0]-alpha_3)**2+(parameters[2]-sigma)**2+(parameters[1]-alpha_1)**2))
    error_data[_,:]=np.array(errors)
    print(_)

ax1.boxplot(error_data)

ax1.grid()
ax1.set_xticklabels([str(i) for i in ['standard', '1/2n', '$1/n^{2}$']])
ax1.set(xlabel='L', ylabel='$\||\Delta x\||_{2}$')
plt.savefig(f'Weighted_large(14)_({alpha_3},{alpha_1},{sigma}).svg')

plt.show()
'''
error_data=[]
kond_number_data=np.array([])
for _ in range(60):
    X = simulation(delta_time, J, 0)
    j = J
    kond_number=[]
    for n_lines in range(3, 10):
        max_degree = n_lines+2
        moments = Moment(max_degree, X, j, delta_time)
        M = Matrix_special(X, n_lines, moments)

        kond_number.append(np.sqrt(LA.cond(np.transpose(M)@M))  )


    print(_)
    kond_number_data=np.append(kond_number_data,kond_number,axis=0)

kond_number_data= kond_number_data.reshape((60,-1))
conditional_numbrs=[np.mean(kond_number_data[:,i]) for i in range(len(range(3,10)))]

fig = plt.figure(figsize=(15, 5))

ax2=fig.add_subplot(111)
ax2.scatter(np.array(range(3,10)),kond_number_data)
ax2.grid()
ax2.set(xlabel='L ', ylabel='Conditional number')
plt.savefig(f'Cond_number_error({alpha_3},{alpha_1},{sigma}).svg')
plt.show()
'''