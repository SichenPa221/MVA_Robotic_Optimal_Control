'''
iLQR applied to a bicopter for a viapoints task (batch formulation)

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt

# computer the transfer matrix of the linearized system
def transferMatrices(A, B):
    nbVarX, nbVarU, nbData = B.shape
    nbData += 1
    Sx = np.kron(np.ones((nbData, 1)), np.identity(nbVarX))
    Su = np.zeros((nbVarX * (nbData), nbVarU * (nbData-1)))
    for t in range(nbData-1):
        id1 = np.arange(t*nbVarX, (t+1)*nbVarX, 1, dtype=int) # 012, 345, ...
        id2 = np.arange((t+1)*nbVarX, (t+2)*nbVarX, 1, dtype=int) # 345, 678, ...
        id3 = np.arange(t*nbVarU, (t+1)*nbVarU, 1, dtype=int) # 012, 345, ...
        Sx[id2, :] = np.matmul(A[:, :, t], Sx[id1, :])
        Su[id2, :] = np.matmul(A[:, :, t], Su[id1, :])
        Su[(t+1)*nbVarX : (t+2)*nbVarX, t*nbVarU : (t+1)*nbVarU] = B[:, :, t]
    return Su, Sx

# Given the control trajectory u and initial state x0, compute the whole state trajectory
def dynSysSimulation(x0, u, model):
    x = np.zeros([model.nbVarX, model.nbData])
    dx = np.zeros(param.nbVarX)
    x[:,0] = x0
    for t in range(param.nbData-1):
        dx[:3] = x[3:,t]
        dx[3] = -(u[0,t] + u[1,t]) * np.sin(x[2,t]) / param.m
        dx[4] =  (u[0,t] + u[1,t]) * np.cos(x[2,t]) / param.m - param.g
        dx[5] =  (u[0,t] - u[1,t]) * param.l / param.I 
        x[:,t+1] = x[:,t] + dx * param.dt
    return x

# Linearize the system along the trajectory computing the matrices A and B
def linSys(x, u, param):
    A = np.zeros([param.nbVarX, param.nbVarX, param.nbData-1])
    B = np.zeros([param.nbVarX, param.nbVarU, param.nbData-1])
    Ac = np.zeros([param.nbVarX, param.nbVarX])
    Ac[:3,3:] = np.eye(param.nbVarPos)
    Bc = np.zeros([param.nbVarX, param.nbVarU])
    for t in range(param.nbData-1):
        # Linearize the system
        Ac[3,2] = -(u[0,t] + u[1,t]) * np.cos(x[2,t]) / param.m
        Ac[4,2] = -(u[0,t] + u[1,t]) * np.sin(x[2,t]) / param.m
        Bc[3,0] = -np.sin(x[2,t]) / param.m
        Bc[3,1] =  Bc[3,0]
        Bc[4,0] =  np.cos(x[2,t]) / param.m
        Bc[4,1] =  Bc[4,0]
        Bc[5,0] =  param.l / param.I 
        Bc[5,1] = -Bc[5,0]
        # Discretize the linear system
        A[:,:,t] = np.eye(param.nbVarX) + Ac * param.dt
        B[:,:,t] = Bc * param.dt
    return A, B

# Plot the bicopter at the desired position
def plot_bicopter(x, param, col='black', col_alpha=1):
	offset = 0.5 * param.l * np.array([np.cos(x[2]), np.sin(x[2])])
	xl = x[:2] - offset
	xr = x[:2] + offset
	plt.plot([xl[0], xr[0]], [xl[1], xr[1]], linewidth=2, c=col, alpha=col_alpha)
	plt.scatter(x[0], x[1], marker='.', s=200, c=col, alpha=col_alpha)

## Parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.dt = 5E-2 # Time step length
param.nbData = 100 # Number of datapoints
param.nbIter = 100 # Maximum number of iterations for iLQR
param.nbPoints = 1 # Number of viapoints
param.nbVarPos = 3 # Dimension of position (x1,x2,theta)
param.nbDeriv = 2 # Number of derivatives (nbDeriv=2 for [x; dx] state)
param.nbVarX = param.nbVarPos * param.nbDeriv # State space dimension
param.nbVarU = 2 # Control space dimension (u1,u2)

param.l = 0.5 # Length of the bicopter
param.m = 1.5 # Mass of the bicopter
#param.I = 1 # Inertia
#param.I = param.m * param.l**2 / 12 # Inertia (homogeneous tube of length l)
param.I = 2. * param.m * param.l**2 # Inertia (two masses at distance l)
param.g = 9.81 # Acceleration due to gravity
param.q = 1E3 # Precision weight
param.r = 1E-3 # Control weight term
param.Mu = np.array([4., 4., 0, 0, 0, 0]) # Viapoints 

Q = np.eye(param.nbVarX * param.nbPoints) * param.q # Precision matrix
R = np.eye((param.nbData-1) * param.nbVarU) * param.r # Control weight matrix

# Time occurrence of viapoints
tl = np.linspace(0, param.nbData, param.nbPoints+1)
tl = np.rint(tl[1:]).astype(np.int64) - 1
idx = np.array([i + np.arange(0,param.nbVarX,1) for i in (tl*param.nbVarX)]).flatten() 


# Solving iLQR
# ===============================

u = np.zeros(param.nbVarU*(param.nbData-1)) # Initial control command
x0 = np.zeros(param.nbVarX) # Initial state

for i in range(param.nbIter):
	# System evolution
    x = dynSysSimulation(x0, u.reshape((param.nbData-1, param.nbVarU)).T, param)

    # Linearization
    A, B = linSys(x, u.reshape((param.nbData-1, param.nbVarU)).T, param)
    Su0, _ = transferMatrices(A, B)
    Su = Su0[idx,:]
    
    # Gauss-Newton update
    e = x[:,tl].flatten() - param.Mu 
    du = np.linalg.inv(Su.T @ Q @ Su + R) @ (-Su.T @ Q @ e - R @ u)
    
    # Estimate step size with backtracking line search method
    alpha = 1
    cost0 = e.T @ Q @ e + u.T @ R @ u
    while True:
        utmp = u + du * alpha
        xtmp = dynSysSimulation(x0, utmp.reshape((param.nbData-1, param.nbVarU)).T, param)
        etmp = xtmp[:,tl].flatten() - param.Mu
        cost = etmp.T @ Q @ etmp + utmp.T @ R @ utmp
        
        if cost < cost0 or alpha < 1e-3:
            u = utmp
            print("Iteration {}, cost: {}, alpha: {}".format(i, cost, alpha))
            break
        alpha /= 2
    
    if np.linalg.norm(alpha * du) < 1e-2: # Early stop condition
        break

# Plotting
# ===============================
plt.figure()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

plt.plot(x[0,:], x[1,:], c='black')

nb_plots = 15
for i in range(nb_plots):
  plot_bicopter(x[:,int(i*param.nbData/nb_plots)], param, 'black', 0.1+0.9*i/nb_plots)
plot_bicopter(x[:,-1], param, 'black')

plot_bicopter(param.Mu, param, 'red')
plt.scatter(param.Mu[0], param.Mu[1], color='r', marker='.', s=200, label="Desired pose")
plt.legend()

plt.show()

