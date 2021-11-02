from matplotlib.markers import MarkerStyle
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.numeric import allclose
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib.animation import FuncAnimation 
import pylab as p

def eqn(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (float(alpha) - float(beta) * y)
    doty = y * (-float(gamma)+ float(delta) * x)
    return np.array([dotx, doty])

def Euler(func, X0, t, alpha, beta, delta, gamma):
    dt = t[1] - t[0]
    nt = len(t)
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        X[i+1] = X[i] + func(X[i], t[i], alpha,  beta, delta, gamma) * dt
    return X

def RK2(func, X0, t, alpha, beta, delta, gamma):
    dt = t[1] - t[0]
    nt = len(t)
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 =dt* func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = dt*func(X[i] +  k1, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + (k1 +  k2 )/2
    return X

def RK4(func, X0, t, alpha,  beta, delta, gamma):
    dt = t[1] - t[0]
    nt = len(t)
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i], alpha,  beta, delta, gamma)
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2., alpha,  beta, delta, gamma)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2., alpha,  beta, delta, gamma)
        k4 = func(X[i] + dt    * k3, t[i] + dt, alpha,  beta, delta, gamma)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X

alpha = 1.1;beta = 0.4;delta = 0.1;gamma = 0.4;x0 = 10;y0 = 10;Nt = 1000000;tmax = 100
t = np.linspace(0.,tmax, Nt)
X0 = [x0, y0]
res = integrate.odeint(eqn, X0, t, args = (alpha, beta, delta, gamma))
x, y = res.T
data = {"t":t,"x":x,"y":y}
print(pd.DataFrame(data))
#plt.figure()
plt.grid()
plt.title("odeint method")
plt.plot(t, x, '-', label = 'Baboon')
plt.plot(t, y, '-', label = "Cheetah")
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
plt.legend()
#plt.show()

Xe = Euler(eqn, X0, t, alpha, beta, delta, gamma)
x1,y1=Xe.T
data = {"x":x1,"y":y1}
print(pd.DataFrame(data))
#plt.figure()
plt.title("Euler method")
plt.plot(t, Xe[:, 0], '-', label = 'Baboon')
plt.plot(t, Xe[:, 1], '-', label = "Cheetah")
plt.grid()
plt.xlabel("Time, $t$ [s]")
plt.ylabel('Population')
plt.legend(loc = "best")
#plt.show()


Xrk2 = RK2(eqn, X0, t, alpha,  beta, delta, gamma)
x3,y3=Xrk2.T
data = {"x":x3,"y":y3}
print(pd.DataFrame(data))
#plt.figure()
plt.title("RK2 method")
plt.plot(t, Xrk2[:, 0], '-', label = 'Baboon')
plt.plot(t, Xrk2[:, 1], '-', label = "Cheetah")
plt.grid()
plt.xlabel("Time, $t$ [s]")
plt.ylabel('Population')
plt.legend(loc = "best")
#plt.show()


Xrk4 = RK4(eqn, X0, t, alpha,  beta, delta, gamma)
x2,y2=Xrk4.T
data = {"x":x2,"y":y2}
print(pd.DataFrame(data))
#plt.figure()
plt.title("RK4 method")
plt.plot(t, Xrk4[:, 0], '-', label = 'Baboon')
plt.plot(t, Xrk4[:, 1], '-', label = "Cheetah")
plt.grid()
plt.xlabel("Time, $t$ [s]")
plt.ylabel('Population')
plt.legend(loc = "best")
plt.show()

# equilibrium
X_f0 = np.array([     0 ,  0])
X_f1 = np.array([ int(gamma/delta), int(alpha/beta)])
d=np.zeros(2)
print(np.allclose(eqn(X_f0,t,alpha,beta,delta,gamma),d))
print(np.allclose(eqn(X_f1,t,alpha,beta,delta,gamma),d))

#jacobian
def jacob(X, t, alpha, beta, delta, gamma):
    x, y = X
    return np.array([[alpha -beta*y,   -beta*x     ],
                  [delta*y ,   delta*x-gamma] ])

er=np.array([gamma/delta,alpha/beta])

A_f1=jacob(er,t,alpha,beta,delta,gamma)  #oscillation
lambda1, lambda2 = np.linalg.eigvals(A_f1) 
print(lambda1,lambda2)
T_f1 = 2*np.pi/(lambda1*lambda2)
print(T_f1)

A_f1=jacob(X_f0,t,alpha,beta,delta,gamma)  #stationary
lambda1, lambda2 = np.linalg.eigvals(A_f1)  
print(lambda1,lambda2)
T_f1 = 2*np.pi/(lambda1*lambda2)
print(T_f1)


'''
fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

data_skip = 50


def init_func():
    ax1.clear()
    ax2.clear()
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax2.set_xlabel('t')
    ax2.set_ylabel('y')
    ax1.set_xlim((t[0], t[-1]))
    ax1.set_ylim((0, 30))
    ax2.set_xlim((t[0], t[-1]))
    ax2.set_ylim((0, 30))


def update_plot(i):
    # ax.clear()
    ax1.plot(t[i:i+data_skip], x2[i:i+data_skip], color='k')
    ax1.scatter(t[i], x2[i], marker='o', color='r')
    ax2.plot(t[i:i+data_skip], y2[i:i+data_skip], color='k')
    ax2.scatter(t[i], y2[i], marker='o', color='r')


anim = FuncAnimation(fig,
                     update_plot,
                     frames=np.arange(0, len(t), data_skip),
                     init_func=init_func,
                     interval=1)

anim.save('animationbb.gif', dpi=150, fps=10, writer='ffmpeg')
'''
values  = np.arange(1, 9, 1)                          # position of X0 between X_f0 and X_f1
vcolors = plt.cm.autumn_r(np.linspace(0.1, 0.9, len(values)))  # colors for each trajectory
h1=[];h2=[]
for v, col in zip(values, vcolors):
    X0 = v * X_f1                             # starting point
    X = integrate.odeint(eqn, X0, t, args = (alpha, beta, delta, gamma))       # we don't need infodict here
    plt.plot( X[:,0], X[:,1], lw=3.5, color=col, label='X0=(%.f, %.f)' % ( X0[0], X0[1]) )
    plt.legend()
    h1.append(X0[0])
    h2.append(X0[1])

ymax = plt.ylim(ymin=0)[1]                    # get axis limits
xmax = plt.xlim(xmin=0)[1]
nb_points   = 30

x = np.linspace(0, xmax, nb_points)
y = np.linspace(0, ymax, nb_points)

X1 , Y1  = p.meshgrid(x, y)    
                # create a grid
DX1, DY1 = eqn([X1,Y1], t, alpha, beta, delta, gamma)               # compute growth rate on the gridt
M = (np.hypot(DX1, DY1))                           # Norm of the growth rate 
M[ M == 0] = 1.                             # Avoid zero division errors 
DX1 /= M                                        # Normalize each arrows
DY1 /= M

#plt.scatter([0,alpha/beta],[0,delta/gamma])
plt.title('Trajectories and direction fields')
plt.scatter(h1,h2,color="black",marker="*",s=200)
Q = plt.quiver(X1, Y1, DX1, DY1, M, pivot='mid', cmap=p.cm.jet)
plt.xlabel('Number of baboon')
plt.ylabel('Number of cheetah')
plt.legend()
plt.grid()
plt.xlim(0, xmax)
plt.ylim(0, ymax)
plt.show()

from mpl_toolkits import mplot3d
'''
x = np.outer(x2, np.ones(1000))
y = np.outer( np.ones(1000),y2)
z = np.outer(t)
print(shape(z))
'''
'''
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(x2,y2,t)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("t")
plt.show()
'''
def V(x,y, alpha, beta, delta, gamma):
    return np.exp(-(delta*x -gamma*np.log(x)+beta*y-alpha*np.log(y)))


x = np.outer(x2, np.ones(Nt))
y = np.outer( np.ones(Nt),y2)
z = V(x,y,alpha,beta,delta,gamma)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("V")
plt.show()

'''
#!python
#-------------------------------------------------------
# plot iso contours
nb_points = 80                            
x = np.linspace(0, xmax, nb_points)
y = np.linspace(0, ymax, nb_points)
X2 , Y2  = np.meshgrid(x, y)                  
Z2 = V(X2,Y2,alpha,beta,delta,gamma)                          
f3 = p.figure()
CS = p.contourf(X2, Y2, Z2, cmap=p.cm.Purples_r, alpha=0.5)
CS2 = p.contour(X2, Y2, Z2, colors='black', linewidths=2. )
p.clabel(CS2, inline=1, fontsize=16, fmt='%.f')
p.grid()
p.xlabel('Number of rabbits')
p.ylabel('Number of foxes')
p.ylim(1, ymax)
p.xlim(1, xmax)
p.title('IF contours')
p.show()
'''

for v in values:
    X0 = v * X_f1                               # starting point
    X = integrate.odeint(eqn, X0, t, args = (alpha, beta, delta, gamma))
    x,y=X.T
    I = V(x,y,alpha,beta,delta,gamma)                                 # compute IF along the trajectory
    #print(I)
    #I_mean = I.mean()
    #delta = 100 * (I.max()-I.min())/I_mean
    #print('X0=(%2.f,%2.f) => I ~ %.1f |delta = %.3G %%' % (X0[0], X0[1], I_mean, delta))
I = V(x2,y2,alpha,beta,delta,gamma) 
I_m=(np.mean(I)/1000)*len(I)
print(I)
data={"I":I,"I_m":I_m}
print(pd.DataFrame(data))
print(np.allclose(I,I_m,rtol=1e-3))