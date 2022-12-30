
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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



T=30000
delta_time=0.01
J=int(T/delta_time)



fig = plt.figure(figsize=(10, 10))

ax1=fig.add_subplot(111)
error_data=np.zeros(shape=(10,len(range(3000,J,2000))))

for n_lines in range(3,4):
    for _ in range(10):
        errors = []
        X = simulation(delta_time, J, 0)
        for j in range(3000,J,2000):
            max_degree = n_lines
            moments = Moment(max_degree, X, j, delta_time)
            M = Matrix_special(X, n_lines, moments)
            parameters=finding_parameters(X,j,delta_time,M)
            errors.append(np.sqrt((parameters[0]-alpha)**2+(parameters[1]-sigma)**2))
        error_data[_,:]=errors

avarage_error=np.zeros(len(range(3000,J,2000)))
for _ in range(10):
    avarage_error=avarage_error+error_data[_,:]

x=np.log(delta_time*np.array(range(3000,J,2000)))
y=np.log(avarage_error)
ax1.plot(x,y,label='Actual error')
m,b = np.polyfit(x, y, 1)
ax1.plot(x, -0.5*x + b * np.ones(len(x)), label='Line with slope coef.:'+f'{"{:.2f}".format(-0.5)}')
ax1.grid()

ax1.set(xlabel='log(T) ', ylabel='log(Error)')
ax1.legend(loc='center right',prop={'size': 10}, bbox_to_anchor=(1, 0.5))
plt.savefig(f'errorVStimeOU({alpha},{sigma}).svg')
print(b,m)

plt.show()