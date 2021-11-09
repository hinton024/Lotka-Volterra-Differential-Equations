import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib.animation import FuncAnimation 
import pylab as p

def eqn(t,X, cons):   #Defining Equations 
    x, y = X          #Assigning x,y values 
    alpha,beta,delta,gamma=cons       #Assigning values to parameters 
    dotx = x * (float(alpha) - float(beta) * y)       #dx/dt = alpha*x-beta*x*y
    doty = y * (-float(gamma)+ float(delta) * x)      #dy/dt = delta*x*y-gamma*y
    return np.array([dotx, doty])                     #Returns dx/dt and dy/dt in array 

def eqn1(X, t, alpha, beta, delta, gamma):           #same equations , but taking alpha,beta,delta,gamma parameters as input instead of array
    x, y = X                                         
    dotx = x * (float(alpha) - float(beta) * y)
    doty = y * (-float(gamma)+ float(delta) * x)
    return np.array([dotx, doty])

def Euler(func, X0, tmin,tmax,N, cons):    #Euler Method
    t = np.linspace(tmin,tmax, N)          #Creating array containg N-2 points between tmin and tmax (Basically calculation points)
    dt = t[1] - t[0]                       # calculating step size
    X  = np.zeros([N, len(X0)])            #creating dummy for output array containing x,y values
    X[0] = X0                              #assigning initial values to the output array
    for i in range(N-1):
        X[i+1] = X[i] + func(t[i],X[i],cons) * dt          #Updating values of the dummy array that we created 
    return X,t                                             #returns array of updated values of X and array t

def RK2(func, X0, tmin, tmax,N,cons):         #Runga kutta 2 method
    t = np.linspace(tmin,tmax, N)   #Creating array containg N-2 points between tmin and tmax (Basically calculation points)
    dt = t[1] - t[0]                 # calculating step size
    X  = np.zeros([N, len(X0)])      #creating dummy for output array containing x,y values
    X[0] = X0                        #assigning initial values to the output array
    for i in range(N-1):
        k1 =dt* func(t[i], X[i], cons)
        k2 = dt*func(t[i] + dt,X[i] +  k1 , cons)
        X[i+1] = X[i] + (k1 +  k2 )/2                           #Updating values of the dummy array that we created 
    return X,t                                     #returns array of updated values of X and array t

def RK4(func, X0, tmin, tmax,N,cons):         #Runga kutta 4 method
    t = np.linspace(tmin,tmax, N)   #Creating array containg N-2 points between tmin and tmax (Basically calculation points)
    dt = t[1] - t[0]                 # calculating step size
    X  = np.zeros([N, len(X0)])      #creating dummy for output array containing x,y values
    X[0] = X0                        #assigning initial values to the output array
    for i in range(N-1):
        k1 = func( t[i],X[i], cons)
        k2 = func(t[i] + dt/2.,X[i] + dt/2. * k1, cons)
        k3 = func( t[i] + dt/2., X[i] + dt/2. * k2,cons)
        k4 = func( t[i] + dt, X[i] + dt    * k3,cons)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)           #Updating values of the dummy array that we created 
    return X,t                                                       #returns array of updated values of X and array t

def V(X,cons):      #Analytic solution of the equation V = delta*x - gamma*ln(x)+beta*y-alpha*ln(y)   
    x,y=X
    alpha,beta,delta,gamma=cons
    return delta*x -gamma*np.log(x)+beta*y-alpha*np.log(y)

def main():
    t = np.linspace(0.,tmax, Nt)      #Creating array containg N-2 points between tmin and tmax (Basically calculation points)
    X0 = [x0, y0]       #Initial conditions
    cons=(alpha,beta,delta,gamma)
    res = integrate.odeint(eqn1, X0, t, args = cons)    #Calling Scipy's odeint function
    x, y = res.T
    Xe = Euler(eqn, X0,0.,tmax,Nt,cons)    #calling Euler
    x1,y1=Xe[0].T
    Xr2 = RK2(eqn, X0,0.,tmax,Nt,cons)      #calling RK2
    x2,y2=Xr2[0].T
    Xr4 = RK4(eqn, X0,0.,tmax,Nt,cons)       #calling RK4
    x3,y3=Xr4[0].T

    xee=[];yee=[];xer2=[];yer2=[];xer4=[];yer4=[]               #Creating empty lists for appending values of abs(odeint-numrical method(euler/rk2/rk4))/odeint
    for i in range(len(x)):                               #comparing our results of Euler,RK2,RK4 with Scipy's ODEint for finding error
        xee.append((x[i]-x1[i])/x[i])
        yee.append((y[i]-y1[i])/y[i])
        xer2.append((x[i]-x2[i])/x[i])
        yer2.append((y[i]-y2[i])/y[i])
        xer4.append((x[i]-x3[i])/x[i])
        yer4.append((y[i]-y3[i])/y[i])              #appending values 
    V_data = V(X=[x3,y3],cons=cons)
    return (t,x,x1,x2,x3,y,y1,y2,y3,xee,yee,xer2,yer2,xer4,yer4,V_data) 

x0 = 30;y0 = 4        # (*1000) INITIAL CONDITIONS     
alpha = 0.453;beta = 0.0205;gamma = 0.790;delta = 0.0229;tmax = 100        #Assigning values of parameters
N_arr=[1000,10000,100000,1000000]                #array for different number of steps

for Nt in N_arr:                                 #calculations for different N and storing values in csv files
    k=main()
    t,x,x1,x2,x3,y,y1,y2,y3,xee,yee,xer2,yer2,xer4,yer4,V_data=k
    DataOut111 = np.column_stack((t,x,x1,x2,x3,y,y1,y2,y3,xee,yee,xer2,yer2,xer4,yer4,V_data))
    if Nt==1000:
        np.savetxt('data_1000.csv', DataOut111,delimiter=',') 
    elif Nt==10000:
        np.savetxt('data_10000.csv', DataOut111,delimiter=',') 
    elif Nt==100000:
        np.savetxt('data_100000.csv', DataOut111,delimiter=',') 
    elif Nt==1000000:
        np.savetxt('data_1000000.csv', DataOut111,delimiter=',') 

Nt=10000
G=main()
t,x,x1,x2,x3,y,y1,y2,y3,xee,yee,xer2,yer2,xer4,yer4,V_data=G
print("FOR N = ",Nt)
print("----------------------TABLE SHOWING VARIATION OF POPULATION OF PREY(x) WITH TIME(t)-------------------------")
data={"t":t,"x(ODEINT)*1000":x ,"x(Euler)*1000":x1,"x(RK2)*1000":x2,"x(RK4)*1000":x3,"[x(odeint)-x(euler)]/x(odeint)":xee,"[x(odeint)-x(rk2)]/x(odeint)":xer2,"[x(odeint)-x(rk4)]/x(odeint)":xer4}
print(pd.DataFrame(data))
print("--------------------TABLE SHOWING VARIATION OF POPULATION OF PREDATOR(y) WITH TIME(t)-----------------------")
data={"t":t,"y(ODEINT)*1000":y ,"y(Euler)*1000":y1,"y(RK2)*1000":y2,"y(RK4)*1000":y3,"[y(odeint)-y(euler)]/y(odeint)":yee,"[y(odeint)-y(rk2)]/y(odeint)":yer2,"[y(odeint)-y(rk4)]/y(odeint)":yer4}
print(pd.DataFrame(data))

X_e1 = np.array([     0 ,  0])                   #Equilibrium condition 1
X_e2 = np.array([ gamma/delta, alpha/beta])      #Equilibrium condition 2                                   
print("GROWTH RATE AT EQUILIBRIUM POINTS FOUND ANALYTICALLY")
print("At (0,0), Growth Rate = ",eqn1(X_e1,t,alpha,beta,delta,gamma))
print("At (gamma/delta,alpha/beta), Growth Rate = ",eqn1(X_e2,t,alpha,beta,delta,gamma))

#Plotting Trajectories and direction fields
values  = np.linspace(1,5, 5)                          
X_f1 = np.array([ int(gamma/delta), int(alpha/beta)])
vcolors = plt.cm.autumn_r(np.linspace(0.1, 0.9, len(values)))  # colors for each trajectory
h1=[];h2=[]
for v, col in zip(values, vcolors):
    X0 = v * X_f1                             # starting point
    X = integrate.odeint(eqn1, X0, t, args = (alpha, beta, delta, gamma))     
    plt.plot( X[:,0], X[:,1], lw=3.5, color=col, label='X0=(%.f, %.f)' % ( X0[0], X0[1]) )
    plt.legend()
    h1.append(X0[0])
    h2.append(X0[1])

ymax = plt.ylim(ymin=0)[1]                    # get axis limits
xmax = plt.xlim(xmin=0)[1]
nb_points   = 30

x = np.linspace(0, xmax, nb_points)
y = np.linspace(0, ymax, nb_points)

X1 , Y1  = p.meshgrid(x, y)    # create a grid     
DX1, DY1 = eqn1([X1,Y1], t, alpha, beta, delta, gamma)               # compute growth rate on the grid
M = (np.hypot(DX1, DY1))                           # Norm of the growth rate 
M[ M == 0] = 1.                              # Avoid zero division errors 
DX1 /= M                                        # Normalize each arrows
DY1 /= M

plt.title('Trajectories and direction fields',fontsize=22,c="r")
plt.scatter(h1,h2,color="black",marker="*",s=300,label="Initial Conditions")
Q = plt.quiver(X1, Y1, DX1, DY1, M, pivot='mid', cmap=p.cm.jet)
plt.xlabel('Number of prey (*1000)',fontsize=19,c="green")
plt.ylabel('Number of predator (*1000)',fontsize=19,c="green")
plt.legend()
plt.grid()
plt.xlim(0, xmax)
plt.ylim(0, ymax)
plt.show()

#ANIMATION
fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
data_skip = 200

def init_func():
    ax1.clear()
    ax2.clear()
    ax1.set_xlabel('Time ')
    ax1.set_ylabel('[Prey(red) & Predator(green)]*1000')
    ax2.set_xlabel('Prey (*1000)')
    ax2.set_ylabel('Predator (*1000)')
    ax1.set_xlim((t[0], t[-1]))
    ax1.set_ylim((0, 85))
    ax1.grid()
    ax2.set_xlim((5,85))
    ax2.set_ylim((0,70))
    ax2.grid()

def update_plot(i):
    ax1.plot(t[i:i+data_skip], x3[i:i+data_skip], color='k')
    ax1.scatter(t[i], x3[i], marker='o', color='r',label="prey")
    ax1.plot(t[i:i+data_skip], y3[i:i+data_skip], color='k')
    ax1.scatter(t[i], y3[i], marker='o', color='green',label="predator")
    ax2.plot(x3[i:i+data_skip], y3[i:i+data_skip], color='k')
    ax2.scatter(x3[i], y3[i], marker='o', color='magenta')

anim = FuncAnimation(fig,
                     update_plot,
                     frames=np.arange(0, len(t), data_skip),
                     init_func=init_func,
                     interval=1)

anim.save('animation_LotVol.gif', dpi=150, fps=10, writer='ffmpeg')