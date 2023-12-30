'''
iLQR applied to a car for a viapoints task (batch formulation)

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
        dx[0] = np.cos(x[2,t]) * u[0,t]
        dx[1] = np.sin(x[2,t]) * u[0,t]
        dx[2] = np.tan(x[3,t]) * u[0,t] / param.l 
        dx[3] = u[1,t] 
        x[:,t+1] = x[:,t] + dx * param.dt
    return x

# Linearize the system along the trajectory computing the matrices A and B
def linSys(x, u, param):
    A = np.zeros([param.nbVarX, param.nbVarX, param.nbData-1])
    B = np.zeros([param.nbVarX, param.nbVarU, param.nbData-1])
    Ac = np.zeros([param.nbVarX, param.nbVarX])
    Bc = np.zeros([param.nbVarX, param.nbVarU])
    for t in range(param.nbData-1):
        # Linearize the system
        Ac[0,2] = -u[0,t] * np.sin(x[2,t])
        Ac[1,2] = u[0,t] * np.cos(x[2,t])
        Ac[2,3] = u[0,t] * np.tan(x[3,t]**2+1) / param.l
        Bc[0,0] = np.cos(x[2,t])  
        Bc[1,0] = np.sin(x[2,t])
        Bc[2,0] = np.tan(x[3,t]) / param.l
        Bc[3,1] = 1 
        # Discretize the linear system
        A[:,:,t] = np.eye(param.nbVarX) + Ac * param.dt
        B[:,:,t] = Bc * param.dt
    return A, B

# Plot the car at the desired position
def plot_car(x, param, col='black', col_alpha=1):
	w = param.l / 2
	x_rl = x[:2] + 0.5 * w * np.array([-np.sin(x[2]), np.cos(x[2])])
	x_rr = x[:2] - 0.5 * w * np.array([-np.sin(x[2]), np.cos(x[2])])
	x_fl = x_rl + param.l * np.array([np.cos(x[2]), np.sin(x[2])])
	x_fr = x_rr + param.l * np.array([np.cos(x[2]), np.sin(x[2])])
	x_plot = np.concatenate((x_rl, x_rr, x_fr, x_fl, x_rl))
	plt.plot(x_plot[0::2], x_plot[1::2], linewidth=2, c=col, alpha=col_alpha)
	plt.scatter(x[0], x[1], marker='.', s=200, c=col, alpha=col_alpha)

## Parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.dt = 1E-1 # Time step length
param.nbData = 50 # Number of datapoints
param.nbIter = 20 # Maximum number of iterations for iLQR
param.nbPoints = 1 # Number of viapoints
param.nbVarX = 4 # State space dimension (x1,x2,theta,phi)
param.nbVarU = 2 # Control space dimension (v,dphi)
param.l = .25 # Length of the car
param.q = 1E3 # Precision weight
param.r = 1E-3 # Control weight term
param.Mu = np.array([2., 1., 0, 0]) # Viapoint (x1,x2,theta,phi)

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
    x = dynSysSimulation(x0, u.reshape([param.nbVarU, param.nbData-1], order='F'), param)

    # Linearization
    A, B = linSys(x, u.reshape([param.nbVarU, param.nbData-1], order='F'), param)
    Su0, _ = transferMatrices(A, B)
    Su = Su0[idx,:]
    
    # Gauss-Newton update
    e = x[:,tl].flatten('F') - param.Mu.flatten('F') 
    du = np.linalg.inv(Su.T @ Q @ Su + R) @ (-Su.T @ Q @ e - R @ u)
    
    # Estimate step size with backtracking line search method
    alpha = 1
    cost0 = e.T @ Q @ e + u.T @ R @ u
    while True:
        utmp = u + du * alpha
        xtmp = dynSysSimulation(x0, utmp.reshape([param.nbVarU, param.nbData-1], order='F'), param)
        etmp = xtmp[:,tl].flatten('F') - param.Mu.flatten('F')
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
  plot_car(x[:,int(i*param.nbData/nb_plots)], param, 'black', 0.1+0.9*i/nb_plots)
plot_car(x[:,-1], param, 'black')

plot_car(param.Mu, param, 'red')
plt.scatter(param.Mu[0], param.Mu[1], color='r', marker='.', s=200, label="Desired pose")
plt.legend()

plt.show()

