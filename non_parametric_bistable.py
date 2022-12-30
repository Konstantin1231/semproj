import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
alpha_3=-1
alpha_2=0
alpha_1=1
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
        X_new=X_old +(alpha_3*(X_old**3)+ alpha_1*X_old)*delta_time+ np.sqrt(2*sigma*delta_time)*norm.rvs()
        sim.append(X_new)
    return sim



''' 
Estimeteurs
'''
def Moment(max_degree,X,J):
    moments=np.ones(max_degree+1)
    V = np.array(X[0:J-1])
    for i in range(1,max_degree+1):
        moments[i]=(1/J) * np.sum(V)
        V=V*np.array(X[0:J-1])
    return moments

def Matrix_special(degree,moments):
    n_lignes=degree+1
    M=np.zeros([n_lignes+1,degree+2])
    firt_array = np.array([moments[degree-i] for i in range(degree+1)])
    firt_array=np.append(firt_array,[0])
    M[0,:]=firt_array

    for i in range(2,n_lignes+1):
         #M[i-1,:]=np.array([moments[i+1],moments[i],moments[i-1],(i-1)*moments[i-2]])
        n_th_array=np.array([moments[i+degree-1-j] for j in range(degree+1)])
        n_th_array=np.append(n_th_array,[(i-1)*moments[i-2]])
        M[i-1,:]=n_th_array
    M[n_lignes,:]=np.append(np.zeros(degree+1),[1])
    return M

def Matrix_special_(X,n_lignes,moments):
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



T=2000
delta_time=0.01
J=int(T/delta_time)
T=J*delta_time
X=simulation(delta_time,J,0)

fig = plt.figure(figsize=(15, 10))
ax1=fig.add_subplot(111)
x = np.linspace(-4, +4, 100)
ax1.plot(x,alpha_3*x**3+ alpha_1 * x, label='True')

for degree in range(3,7):
    n_lignes=degree+2
    max_degree=n_lignes+degree
    moments = Moment(max_degree, X, J)
    M = Matrix_special(degree, moments)
    #M_= Matrix_special_(n_lignes,moments)
    #print(M,M_)
    parameters=finding_parameters(X,J,delta_time,M)
    poly=lambda x: np.dot(parameters[0:len(parameters)-1],[x**(len(parameters)-2-i) for i in range(len(parameters)-1)])
    ax1.plot(x,poly(x),label=f'{degree}')
    print(parameters)


ax1.grid()
ax1.legend(loc='center right', bbox_to_anchor=(1.1, 0.5))
plt.savefig(f'non-parametric_bistable({alpha_3},{alpha_1},{sigma}).svg')
plt.show()