import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
alpha_1=1
alpha_3=-1
sigma_1=1
''' 
Estimeteurs
'''
def Moment(max_degree,X,j):
    moments=np.ones(max_degree+1)
    V = np.array(X[0:j])
    for i in range(1,max_degree+1):
        moments[i]=(1/j) * np.sum(V)
        V=V*np.array(X[0:j])
    return moments
def stoc_int(max_degree,X,J,delta_time):
    moments = np.ones(max_degree + 1)
    dif=np.array(X[1:J]) - np.array(X[0:J-1])
    V=np.array(X[0:J - 1])
    for i in range(1, max_degree + 1):
        moments[i] =  (1 / (J*delta_time)) * np.dot(V,dif)
        V = V * np.array(X[0:J - 1])
    return moments
def sigma_(X,delta_time,J):
    A1=np.array(X[1:J]) - np.array(X[0:J-1])
    return np.sqrt((1/(J*delta_time)) * np.dot(A1,A1))

def alpha_(X,delta_time,J):
    M=Moment(6,X,J)
    B=stoc_int(3,X,J,delta_time)
    alpha_1=(B[1]*M[6]-B[3]*M[4])/(M[2]*M[6]-M[4]**2)
    alpha_3=-(B[1]*M[4]-B[3]*M[2])/(M[2]*M[6]-M[4]**2)
    return [alpha_1,alpha_3]

def Matrix_special(X,n_lignes,moments):
    M=np.zeros([n_lignes+1,3])
    M[0,:]=np.array([moments[3],moments[1],0])
    for i in range(2,n_lignes+1):
        M[i-1,:]=np.array([moments[i+2],moments[i],(i-1)*moments[i-2]])
    M[n_lignes,:]=np.array([0,0,1])
    return M

def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new=X_old +(alpha_3*(X_old**3)+ alpha_1*X_old)*delta_time+ np.sqrt(2*sigma*delta_time)*norm.rvs()
        sim.append(X_new)
    return sim


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
    return reg.coef_



T=10000
delta_time=0.01
J=int(T/delta_time)

X=simulation(delta_time,J,0)

fig = plt.figure(figsize=(15, 10))
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
ax4=fig.add_subplot(224)
n_lines=4

alphas1_c=[]
alphas3_c=[]
sigmas_c=[]
alphas1=[]
alphas3=[]
sigmas=[]
error_MOM=[]
error_classic=[]
for j in range(3000,J,500):
    max_degree = n_lines+2
    moments = Moment(max_degree, X, j)
    M = Matrix_special(X, n_lines, moments)
    parameters=finding_parameters(X,j,delta_time,M)
    sigmas.append(parameters[2])
    alphas1.append(parameters[1])
    alphas3.append(parameters[0])
    sigmas_c.append(sigma_(X, delta_time, j)**2/2)
    alphas1_c.append(alpha_(X,delta_time,j)[0])
    alphas3_c.append(alpha_(X, delta_time, j)[1])
    error_MOM.append(np.sqrt((parameters[0]-alpha_3)**2+(parameters[2]-sigma)**2+(parameters[1]-alpha_1)**2))
    error_classic.append(np.sqrt((alphas1_c[-1]-alpha_1)**2+(sigmas_c[-1]-sigma)**2+(alphas3_c[-1]-alpha_3)**2))

xx = delta_time * np.array(range(3000,J,500))
ax1.plot(xx,sigma*np.ones(len(xx)),color='r',label='True')
ax1.plot(xx,sigmas_c,label='classical')
ax1.plot(xx,sigmas,label='MoM')

ax2.plot(xx,alpha_3*np.ones(len(xx)),color='r',label='True')
ax2.plot(xx,alphas3_c,label='classical')
ax2.plot(xx,alphas3,label='MoM')

ax3.plot(xx,alpha_1*np.ones(len(xx)),color='r',label='True')
ax3.plot(xx,alphas1_c,label='classical')
ax3.plot(xx,alphas1,label='MoM')

ax4.plot(xx,0*np.ones(len(xx)),color='r',label='True')
ax4.plot(xx,error_classic,label='classical')
ax4.plot(xx,error_MOM,label='MoM')




ax1.set(xlabel='time(T) ', ylabel='Sigma')
ax1.grid()
ax2.set(xlabel='time(T) ', ylabel='Alpha_3')
ax2.grid()
ax3.set(xlabel='time(T) ', ylabel='Alpha_1')
ax3.grid()
ax4.set(xlabel='time(T) ', ylabel='Error')
ax4.grid()
ax4.legend(loc='center right', bbox_to_anchor=(1.28, 0.5))

plt.savefig(f'results({alpha_3},{alpha_1},{sigma}).svg')


plt.show()