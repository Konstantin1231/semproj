import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
#condition 2alpha_0>(2sigma_0)**4
alpha_1=-5
sigma_1=2
X_0=1
lamperti=[(alpha_1-2*sigma_1),sigma_1]
cox=np.array([alpha_1,sigma_1])
'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new = X_old +alpha_1 * ( X_old**3 + X_old) * delta_time + np.sqrt(2*sigma_1 * ((X_old**2+1)**2)*delta_time)*norm.rvs()
        sim.append(X_new)
    return sim



''' 
Estimeteurs
'''
def Moment(max_degree,X,J,delta_time,k):
    moments=np.ones(max_degree+1)
    X = np.array(X[0:J])[::k]
    for i in range(1,max_degree+1):
        moments[i]= np.mean(X**i)
    return moments

def Moment_L(max_degree,X,X_L,J,delta_time,k):
    moments=np.ones(max_degree+1)
    V = np.array(X_L[0:J])[::k]
    for i in range(1,max_degree+1):
        moments[i]=np.mean(V)
        V=V*(np.array(X_L[0:J])[::k])
    mixed_moment=np.ones(max_degree+1)
    V = np.array(X[0:J])[::k]
    for i in range(1, max_degree + 1):
        mixed_moment[i] = np.mean(V)
        V = V * (np.array(X_L[0:J])[::k])
    return moments, mixed_moment

def Matrix_special(X,n_lignes,moments):

    M=np.zeros([n_lignes+1,2])
    M[0,:]=np.array([(moments[3]+moments[1]),0])
    for i in range(1,n_lignes):
        M[i,:]=np.array([(moments[i+3]+moments[i+1]),i*(moments[i+3]+2*moments[i+1]+moments[i-1])])
    M[n_lignes,:]=np.array([0,(moments[4]+2*moments[2]+1)])
    return M

def Matrix_special_Lamperti(X,n_lignes,moments,mixed_moments):
    M=np.zeros([n_lignes+1,2])
    M[0,:]=np.array([mixed_moments[1],0])
    for i in range(1,n_lignes):
        M[i,:]=np.array([mixed_moments[i+1], i*(moments[i])])
    M[n_lignes,:]=np.array([0,1])
    return M


def Matrix_special_Lamperti_Taylor(X,n_lignes,moments):
    M=np.zeros([n_lignes+1,3])
    M[0,:]=np.array([moments[0],moments[2],moments[0]])
    for i in range(1,n_lignes):
        M[i,:]=np.array([moments[i],moments[i+2],(i+1)*(moments[i])])
    M[n_lignes,:]=np.array([0,1])
    return M

def Matrix_special_Lamperti_Taylor_2(X,n_lignes,moments):
    M=np.zeros([n_lignes+1,3])
    M[0,:]=np.array([moments[0],moments[2],moments[0]])
    for i in range(1,n_lignes):
        M[i,:]=np.array([moments[i],moments[i+2],(i+1)*(moments[i])])
    M[n_lignes,:]=np.array([0,1])
    return M

'''
Calculating coef
'''
def finding_parameters(X,J,delta_time,M,k):

    A1 = np.array(X[1:J])[::k] - np.array(X[0:J - 1])[::k]
    Q= k*(1 /(2* J*delta_time)) * np.dot(A1, A1)
    Y=np.zeros(np.shape(M)[0])
    Y[len(Y)-1]=Q
    print(Q)
    reg = LinearRegression(fit_intercept=False).fit(M,Y)
    return reg.coef_, Q

k=100
T=500
delta_time=0.001
J=int(T/delta_time)
n_lines=2
n_lines_L=2
X=np.array(simulation(delta_time,J,X_0))
X_L=np.arctan(X)
error=[]
error_L=[]
Q_c=[]
Q_L=[]

fig = plt.figure(figsize=(15, 5))
ax1=fig.add_subplot(111)
xx=delta_time*np.array([int(i*J/(30)) for i in range(3,30,3)])
ax1.plot(delta_time*np.array(np.cumsum(np.ones(len(X)))),X,label='Standard')
ax1.plot(delta_time*np.array(np.cumsum(np.ones(len(X)))),X_L,label='Lamperti')
ax1.set(xlabel='time(T) ', ylabel='$X_{t}$')
ax1.grid()
plt.savefig(f'2X_0_{X_0}_delta_time_{k*delta_time}_PATH({alpha_1},{sigma_1},Time={T}).jpeg')

plt.show()


max_degree = n_lines + 2
for j in [int(i*J/(30)) for i in range(3,30,3)]:

    moments = Moment(max_degree, X, j, delta_time,k)
    #moments_L, mixed_moments_L= Moment_L(max_degree,X, X_L, j, delta_time,k)

    #M = Matrix_special(X, n_lines, moments)
    #M_L = Matrix_special_Lamperti(X_L, n_lines, moments_L,mixed_moments_L)
    M=np.array([[0,0],
                [moments[4]+moments[2],moments[0]+2*moments[2]+moments[4]],
                [0,(moments[4]+2*moments[2]+1)]])
    M_L=[[0,0],
         [np.mean(X[0:j]*X_L[0:j]),1],
         [0,1]
    ]

    parameters, q_c = finding_parameters(X, j, delta_time, M,k)
    parameters_L, q_L=finding_parameters(X_L,j, delta_time, M_L,k)
    parameters_L=[parameters_L[0]+2*parameters_L[1],parameters_L[1]]

    error.append(np.sqrt(np.sum((parameters-cox)**2))/(np.sqrt(np.sum(np.array(cox)**2))))
    error_L.append(np.sqrt(np.sum((parameters_L-cox)**2))/(np.sqrt(np.sum(np.array(cox)**2))))
    Q_c.append(q_c)
    Q_L.append(q_L)


fig = plt.figure(figsize=(15, 5))
ax1=fig.add_subplot(111)

ax1.plot(xx,error)
ax1.plot(xx,error_L)

ax1.set(xlabel='time(T) ', ylabel='Relative Error')
ax1.grid()
plt.savefig(f'2X_0_{X_0}_delta_time_{k*delta_time}_Lamperti_over_time({alpha_1},{sigma_1},Time={T}).svg')
plt.show()


fig = plt.figure(figsize=(15, 5))
ax1=fig.add_subplot(111)
xx=delta_time*np.array([int(i*J/(30)) for i in range(3,30,3)])
ax1.plot(xx,Q_c)
ax1.plot(xx,Q_L)
ax1.set(xlabel='time(T) ', ylabel='$Q_{T}/2T$')
ax1.grid()
plt.savefig(f'2X_0_{X_0}_delta_time_{k*delta_time}_quadratique_estimateur_over_time({alpha_1},{sigma_1},Time={T}).svg')
plt.show()





fig = plt.figure(figsize=(15, 10))
ax1=fig.add_subplot(111)
trials=10
error_data=[]
error_data_L=[]
Q_c=[]
Q_L=[]
j=J
for trial in range(trials):
    X = simulation(delta_time, J, X_0)
    X_L = 2*np.sqrt(np.array(X)+np.ones(len(X)))
    M=np.array([[0,0],
                [moments[4]+moments[2],moments[0]+2*moments[2]+moments[4]],
                [0,(moments[4]+2*moments[2]+1)]])
    M_L=[[0,0],
         [np.mean(X[0:j]*X_L[0:j]),1],
         [0,1]
    ]

    parameters, q_c = finding_parameters(X, j, delta_time, M,k)
    parameters_L, q_L =finding_parameters(X_L,j, delta_time, M_L,k)
    parameters_L = [parameters_L[0] + 2 * parameters_L[1], parameters_L[1]]
    Q_c.append(q_c)
    Q_L.append(q_L)
    error_data.append(np.sqrt(np.sum((parameters-cox)**2))/(np.sqrt(np.sum(np.array(cox)**2))))
    error_data_L.append(np.sqrt(np.sum((parameters_L-cox)**2))/(np.sqrt(np.sum(np.array(cox)**2))))



ax1.boxplot([error_data,error_data_L])
ax1.set_xticklabels(['standard','Lamperti'])
ax1.set( ylabel='Relative Error')
ax1.grid()
plt.savefig(f'2X_0_{X_0}_delta_time_{k*delta_time}_Lamperti_error({alpha_1},{sigma_1},Time={T}).svg')
plt.show()

fig = plt.figure(figsize=(15, 5))
ax1=fig.add_subplot(111)
ax1.boxplot([Q_c,Q_L])
ax1.set_xticklabels(['standard','Lamperti'])
ax1.set( ylabel='$Q_{T}/2T$')
ax1.grid()
plt.savefig(f'2X_0_{X_0}_delta_time_{k*delta_time}_compare_quadratique_estimateur_stat({alpha_1},{sigma_1},Time={T}).svg')
plt.show()
