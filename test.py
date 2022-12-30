import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Parameters of true model
'''
alpha=-1
sigma=1
alpha_0=0.1
X_0=5
'''
Simulation
'''
def simulation(delta_time,J,X_0):
    sim=[]
    sim.append(X_0)
    for _ in range(J):
        X_old=sim[len(sim)-1]
        X_new=X_old +(alpha_0+alpha*X_old)*delta_time+ np.sqrt(2*sigma*delta_time)*norm.rvs()
        sim.append(X_new)
    return sim

def Limits(X):
    X=X[int(0.1*len(X)):]
    X_sorted=np.sort(X)
    lower_index=int(0.025*len(X))
    upper_index=int(0.975*len(X))
    return X_sorted[lower_index],X_sorted[upper_index]




'''
Calculating coef
'''



T=100
delta_time=0.001
J=int(T/delta_time)

X=simulation(delta_time,J,X_0)[0:50000]

fig = plt.figure(figsize=(15, 5))
ax1=fig.add_subplot(111)

[lower,upper]=Limits(X)

xx = delta_time * np.array(range(len(X)))
ax1.plot(xx,X)
ax1.plot(xx,lower*np.ones(len(X)),color='red')
ax1.plot(xx,upper*np.ones(len(X)),color='red')






ax1.set(xlabel='time(T) ', ylabel='$X_{T}$')
ax1.grid()


plt.savefig(f'Limits.jpeg')


plt.show()





