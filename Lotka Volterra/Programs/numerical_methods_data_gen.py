
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.numeric import allclose
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

    xee=[];yee=[];xer2=[];yer2=[];xer4=[];yer4=[]               #Creating empty lists for appending values of absolute error
    for i in range(len(x)):                               #comparing our results of Euler,RK2,RK4 with Scipy's ODEint for finding error
        xee.append((x[i]-x1[i])/x[i])
        yee.append((y[i]-y1[i])/y[i])
        xer2.append((x[i]-x2[i])/x[i])
        yer2.append((y[i]-y2[i])/y[i])
        xer4.append((x[i]-x3[i])/x[i])
        yer4.append((y[i]-y3[i])/y[i])              #appending values of absolute error
    V_data = V(X=[x3,y3],cons=cons)
    return (t,x,x1,x2,x3,y,y1,y2,y3,xee,yee,xer2,yer2,xer4,yer4,V_data)                 

alpha = 1.1;beta = 0.4;delta = 0.1;gamma = 0.4;x0 = 10;y0 = 10;tmax = 100
N_arr=[1000,10000,100000,1000000]
for Nt in N_arr:
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

print("FOR N=100000")
print("----------------------TABLE SHOWING VARIATION OF POPULATION OF PREY(x) WITH TIME(t)-------------------------")
data={"t":t,"x(ODEINT)":x ,"x(Euler)":x1,"x(RK2)":x2,"x(RK4)":x3,"Ab Error (Euler)":xee,"Ab Error (RK2)":xer2,"Ab Error (RK4)":xer4}
print(pd.DataFrame(data))
print("--------------------TABLE SHOWING VARIATION OF POPULATION OF PREDATOR(y) WITH TIME(t)-----------------------")
data={"t":t,"y(ODEINT)":y ,"y(Euler)":y1,"y(RK2)":y2,"y(RK4)":y3,"Ab Error (Euler)":yee,"Ab Error (RK2)":yer2,"Ab Error (RK4)":yer4}
print(pd.DataFrame(data))

X_e1 = np.array([     0 ,  0])                   #Equilibrium condition 1
X_e2 = np.array([ gamma/delta, alpha/beta])      #Equilibrium condition 2 
#d=np.zeros(2)                                    # Creating array [0,0]

print("GROWTH RATE AT EQUILIBRIUM POINTS FOUND ANALYTICALLY")
#prints true if the growth rate is zero
print(eqn1(X_e1,t,alpha,beta,delta,gamma))
print(eqn1(X_e2,t,alpha,beta,delta,gamma))
#print(np.allclose(eqn1(X_e1,t,alpha,beta,delta,gamma),d)) #Checking if the growth rate of predator and prey is zero at equilibrium condition 1
#print(np.allclose(eqn1(X_e2,t,alpha,beta,delta,gamma),d)) #Checking if the growth rate of predator and prey is zero at equilibrium condition 2

