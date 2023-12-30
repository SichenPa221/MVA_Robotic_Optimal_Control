'''
Forward dynamics 

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Amirreza Razmjoo <amirreza.razmjoo@idiap.ch> and 
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://rcfs.ch/>
License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Helper functions
# ===============================

# Forward kinematics for all joints (in robot coordinate system)
def fkin0(x, param): 
	L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
	f = np.vstack([
		L @ np.diag(param.l) @ np.cos(L @ x),
		L @ np.diag(param.l) @ np.sin(L @ x)
	])
	f = np.hstack([np.zeros([2,1]), f])
	return f


# NB: Because of the use of matplotlib.animation, we need to set param as a global variable
# Initialization of the plot
def init():
    global param
    ax.set_xlim(-np.sum(param.l) - 0.1, np.sum(param.l) + 0.1)
    ax.set_ylim(-np.sum(param.l) - 0.1, np.sum(param.l) + 0.1)
    return ln1, ln2

# NB: Because of the use of matplotlib.animation, we need to set param as a global variable
# Updating the values in the plot
def animate(i):
    global param
    f = fkin0(x[:param.nbVarX,i], param)
    ln1.set_data(f[0,:], f[1,:])
    ln2.set_data(f[0,:], f[1,:])
    return ln1, ln2


# General param parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.dt = 1E-2 # Time step length
param.nbData = 20 # Number of datapoints
param.nbVarX = 3 # State space dimension (x1,x2,x3)
param.nbVarU = 3 # Control space dimension (dx1,dx2,dx3)
param.nbVarF = 3 # Objective function dimension (f1,f2,f3, with f3 as orientation)
param.l = np.array([1, 1, 1]) # Robot links lengths
param.m = np.array([1, 1, 1]) # Robot links masses
param.damping = 1 # Viscous friction
param.gravity = 9.81 # Gravity

# Auxiliary matrices
# ===============================
l = np.reshape(param.l, [1, param.nbVarX])
L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
m = np.reshape(param.m, [1, param.nbVarX])
Lm = np.triu(np.ones([m.shape[1], m.shape[1]])) * np.repeat(m, m.shape[1],0)

# Initialization
# ===============================
x = np.zeros([2*param.nbVarX, param.nbData]) # States (position and velocity in the joint space, in matrix form)
tau = np.zeros([param.nbVarU, param.nbData-1]) # Input commands (torque commands, in matrix form)

# Forward Dynamics
for t in range(param.nbData-1):
#	# Elementwise computation of G, C, and M
#    G = np.zeros(param.nbVarX)
#    M = np.zeros([param.nbVarX, param.nbVarX])
#    C =  np.zeros([param.nbVarX, param.nbVarX])
#    for k in range(param.nbVarX):
#        G[k] = -sum(m[0,k:]) * param.gravity * l[0,k] * np.cos(L[k,:] @ x[:param.nbVarX,t])
#        for i in range(param.nbVarX):
#            S = sum(m[0,k:param.nbVarX] * np.heaviside(np.array(range(k, param.nbVarX)) - i, 1))
#            M[k,i] = l[0,k] * l[0,i] * np.cos(L[k,:] @ x[:param.nbVarX,t] - L[i,:] @ x[:param.nbVarX,t]) * S
#            C[k,i] = -l[0,k] * l[0,i] * np.sin(L[k,:] @ x[:param.nbVarX,t] - L[i,:] @ x[:param.nbVarX,t]) * S
	
    # Computation in matrix form of G, C, and M
    G = -np.sum(Lm,1) * param.l * np.cos(L @ x[:param.nbVarX,t]) * param.gravity
    C = -(l.T * l) * np.sin(np.reshape(L @ x[:param.nbVarX,t], [param.nbVarX,1]) - L @ x[:param.nbVarX,t]) * (Lm**.5 @ ((Lm**.5).T))
    M = (l.T * l) * np.cos(np.reshape(L @ x[:param.nbVarX,t], [param.nbVarX,1]) - L @ x[:param.nbVarX,t]) * (Lm**.5 @ ((Lm**.5).T))
    
    G = L.T @ G
    C = L.T @ C 
    M = L.T @ M @ L 
    
    # Compute acceleration
    ddx = np.linalg.inv(M) @ (tau[:,t] + G + C @ (L @ x[param.nbVarX:,t])**2 - x[param.nbVarX:,t] * param.damping)
    
    # compute the next state
    x[:,t+1] = x[:,t] + np.append(x[param.nbVarX:,t], ddx) * param.dt


# Plot
# ===============================
fig, ax = plt.subplots()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')
#xdata, ydata = [], []
ln1, = plt.plot([], [], '-')
ln2, = plt.plot([], [], 'o-', linewidth=2, markersize=5, c="black")
ani = animation.FuncAnimation(fig, animate, x.shape[1], init_func=init, interval = param.dt * 500, blit= True, repeat = False)
plt.show()

