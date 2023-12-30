'''
iLQR determining the optimal time steps evolution to reach a set of viapoints

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Yifei Dong <ydong@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

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
    x[:,0] = x0
    for t in range(param.nbData-1):
        Bd = np.diagflat([u[-1,t], u[-1,t], 1]) # Bd here is different from B in the linearized system
        x[:,t+1] = x[:,t] + Bd @ u[:,t]
    return x

# Linearize the system along the trajectory computing the matrices A and B
def linSys(x, u, param):
    A = np.zeros((param.nbVarX, param.nbVarX, param.nbData-1))
    B = np.zeros((param.nbVarX, param.nbVarU, param.nbData-1))
    for t in range(param.nbData-1):
        A[:, :, t] = np.eye(param.nbVarX)
        B[:, :, t] = np.vstack((
            np.hstack((np.eye(param.nbVarX-1) * u[-1, t], u[0:2, t].reshape(2, 1))), 
            np.array([0, 0, 1])
            ))
    return A, B

# General parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-1 # Time step length
param.nbPoints = 5 # Number of viapoints
param.nbData = 50 # Number of datapoints
param.nbIter = 200 # Maximum number of iterations for iLQR
param.nbVarX = 3 # State space dimension in task space(x1, x2)
param.nbVarU = 3 # Control space dimension in task space(dx1, dx2)
param.Mu = np.vstack((
    np.random.rand(2, param.nbPoints) * 1e2, 
    np.zeros((param.nbVarX-2, param.nbPoints))
    )) # Target viapoints and phase values
# param.Mu = np.array([[-1, 0, 3, 5], [0, 4, 4, 1], [0, 0, 0, 0]])

Q = np.kron(np.eye(param.nbPoints), np.diagflat([1e6, 1e6, 0])) # Precision matrix 

e = np.ones((1, param.nbData-1))
data = np.vstack((-e, 2*e, -e))
diagMat = spdiags(data, np.array([-1, 0, 1]), param.nbData-1, param.nbData-1)
R = np.kron(
    diagMat.toarray(), 
    np.diagflat([1, 1, 1e3])
    ) # Control weight matrix as transition matrix to encourage smoothness

# System parameters
# ===============================

# Time occurrence of viapoints
tl = np.linspace(0, param.nbData-1, param.nbPoints+1)
tl = np.round(tl[1:]).astype(int)
idx = np.array([i + np.arange(0, param.nbVarX, 1, dtype=int) for i in (tl*param.nbVarX)]) 
idx = idx.flatten()

# Solving iLQR
# ===============================

u = np.kron(np.ones((param.nbData-1, 1)), np.array([[0], [0], [param.dt]])) # Initial control command
x0 = np.zeros(param.nbVarX) # Initial state

for i in range(param.nbIter):
	# System evolution
    x = dynSysSimulation(x0, u.reshape((param.nbData-1, param.nbVarU)).T, param)

    # Linearization
    A, B = linSys(x, u.reshape((param.nbData-1, param.nbVarU)).T, param)
    Su0, _ = transferMatrices(A, B)
    Su = Su0[idx, :]
    
    # Increment of control input
    e = x[:, tl] - param.Mu
    e = e.T.reshape((-1, 1))
    du = np.linalg.inv(Su.T @ Q @ Su + R) @ (-Su.T @ Q @ e - R @ u)

    # Perform line search
    alpha = 1
    cost0 = e.T @ Q @ e + u.T @ R @ u
    cost0 = cost0[0, 0]
    
    while True:
        utmp = u + du * alpha
        xtmp = dynSysSimulation(x0, utmp.reshape((param.nbData-1, param.nbVarU)).T, param)
        etmp = xtmp[:, tl] - param.Mu
        etmp = etmp.T.reshape((-1, 1))
        cost = etmp.T @ Q @ etmp + utmp.T @ R @ utmp
        cost = cost[0][0]
        
        if cost < cost0 or alpha < 1e-3:
            u = utmp
            print("Iteration {}, cost: {}, alpha: {}".format(i, cost, alpha))
            break
        alpha /= 2
    
    if np.linalg.norm(alpha * du) < 1e-2: # Early stop condition
        break

# Plotting
# ===============================

# calculate the final state and input trajectory
x = dynSysSimulation(x0, u.reshape((param.nbData-1, param.nbVarU)).T, param)
x = x.T
u = u.reshape((param.nbData-1, param.nbVarU))
t = np.arange(0, param.nbData, 1)

# creating grid for subplots
fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(15)

ax1 = plt.subplot2grid(shape=(3, 4), loc=(0, 0), rowspan=4, colspan=2)
ax2 = plt.subplot2grid(shape=(3, 4), loc=(0, 2))
ax3 = plt.subplot2grid(shape=(3, 4), loc=(0, 3))
ax4 = plt.subplot2grid(shape=(3, 4), loc=(1, 2))
ax5 = plt.subplot2grid(shape=(3, 4), loc=(1, 3))
ax6 = plt.subplot2grid(shape=(3, 4), loc=(2, 2))
ax7 = plt.subplot2grid(shape=(3, 4), loc=(2, 3))

# plotting subplots
# state trajectory in 2D
ax1.scatter(x[0, 0], x[0, 1], c='black', s=40) # start point
ax1.scatter(param.Mu[0], param.Mu[1], c='red', s=60) # target point
ax1.scatter(x[:, 0], x[:, 1], c='black', s=10) # waypoints
ax1.plot(x[:, 0], x[:, 1], c='black') # trajectory
# x1
ax2.plot(t, x[:, 0], 'k')
ax2.scatter(tl, param.Mu[0], c='red', s=60) # target point
ax2.grid(True)
ax2.set_xlabel('t', fontsize=18)
ax2.set_ylabel('$x_1$', fontsize=18)
# u1
ax3.plot(t[:-1], u[:, 0], 'k')
ax3.grid(True)
ax3.set_xlabel('t', fontsize=18)
ax3.set_ylabel('$u_1$', fontsize=18)
# x2
ax4.plot(t, x[:, 1], 'k')
ax4.scatter(tl, param.Mu[1], c='red', s=60) # target point
ax4.grid(True)
ax4.set_xlabel('t', fontsize=18)
ax4.set_ylabel('$x_2$', fontsize=18)
# u2
ax5.plot(t[:-1], u[:, 1], 'k')
ax5.grid(True)
ax5.set_xlabel('t', fontsize=18)
ax5.set_ylabel('$u_2$', fontsize=18)
# s
ax6.plot(t, x[:, 2], 'k')
ax6.grid(True)
ax6.set_xlabel('t', fontsize=18)
ax6.set_ylabel('$s$', fontsize=18)
# us
ax7.plot(t[:-1], u[:, 2], 'k')
ax7.grid(True)
ax7.set_xlabel('t', fontsize=18)
ax7.set_ylabel('$u^s$', fontsize=18)

plt.tight_layout()
plt.show()
