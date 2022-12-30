import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy import linalg as LA
'''
Parameters of true model
'''
alpha=-1
sigma=1
exact_moments=[1,0,1,0,3,0,15,0,105,0]

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
def exacte_Matrix(n_lignes,moments):
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
    reg = LinearRegression(fit_intercept=False).fit(M,Y)
    return reg.coef_

def putting_zeros(moments):
    for i in range(len(moments)):
        if i%2==1 and i>0:
            moments[i]=0
    return moments


T=1000
delta_time=0.01
J=int(T/delta_time)



fig = plt.figure(figsize=(15, 5))

ax1=fig.add_subplot(111)
error_data=[]
kond_number_data=[]
X = simulation(delta_time, J, 0)
for n_lines in range(2,10):
    errors=[]
    kond_number=[]
    for _ in range(1):
        j=J
        max_degree = n_lines
        moments = Moment(max_degree, X, j, delta_time)
        moments = putting_zeros(moments)
        M = Matrix_special(X, n_lines, moments)
        k=np.sqrt(LA.cond(np.transpose(M) @ M))
        kond_number.append(np.sqrt(k))
        parameters=finding_parameters(X,j,delta_time,M)
        errors.append(np.sqrt((parameters[0]-alpha)**2+(parameters[1]-sigma)**2))

    print(moments)
    kond_number_data.append(np.mean(kond_number))
    error_data.append(errors)
print(M)
ax1.boxplot(error_data)

ax1.grid()
ax1.set_xticklabels([str(i) for i in range(2,10)])
ax1.set(xlabel='L', ylabel='$\||\Delta x\||_{2}$')
#plt.savefig(f'error({alpha},{sigma}).svg')



exacte_kond_number=[]
for n_lines in range(2,10):
    M=exacte_Matrix(n_lines,exact_moments)
    k = np.sqrt(LA.cond(np.transpose(M) @ M))
    exacte_kond_number.append(np.sqrt(k))
print(M)

fig = plt.figure(figsize=(15, 5))
ax1=fig.add_subplot(111)
ax1.scatter(np.array(range(2,10)),kond_number_data,label='$ \widehat{M} $')
ax1.scatter(np.array(range(2,10)),exacte_kond_number,label='$M$')
ax1.grid()
ax1.set(xlabel='# of equations (L) ', ylabel='Conditional number K(M)')
ax1.legend(loc='center right',prop={'size': 10}, bbox_to_anchor=(1, 0.5))
plt.savefig(f'Cond_number_EXACTVSAPPROX_zeros({alpha},{sigma}).jpeg')
plt.show()



'''
error_data=[]
kond_number_data=np.array([])
a=100
for _ in range(a):
    X = simulation(delta_time, J, 0)
    j = J
    kond_number=[]
    for n_lines in range(2, 10):
        max_degree = n_lines
        moments = Moment(max_degree, X, j, delta_time)
        M = Matrix_special(X, n_lines, moments)
        parameters = finding_parameters(X, j, delta_time, M)
        kond_number.append(np.sqrt(LA.cond(np.transpose(M)@M))*np.linalg.norm(parameters,ord=2))
        #kond_number.append(np.sqrt(LA.cond(np.transpose(M) @ M)))

    print(_)
    kond_number_data=np.append(kond_number_data,kond_number,axis=0)
    print(kond_number_data)
kond_number_data= kond_number_data.reshape((a,8))
print(kond_number_data)
conditional_numbrs=[np.mean(kond_number_data[:,i]) for i in range(len(range(2,10)))]
fig = plt.figure(figsize=(15, 5))

ax2=fig.add_subplot(111)
ax2.scatter(np.array(range(2,10)),conditional_numbrs)
ax2.grid()
ax2.set(xlabel='L', ylabel='$\kappa(M)\||x\||_{2}$')
plt.savefig(f'Cond_number_error_new_x({alpha},{sigma}).svg')
plt.show()
'''