'''
Linear quadratic tracking (LQT) with a viapoints example

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://rcfs.ch/>
License: MIT
'''

import numpy as np
from math import factorial
import matplotlib.pyplot as plt

# Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.dt = 1E-2 # Time step length
param.nbPoints = 2 # Number of viapoints
param.nbDeriv = 2 # Order of the dynamical system
param.nbVarPos = 2 # Number of position variable
param.nbVar = param.nbVarPos * param.nbDeriv # Dimension of state vector
param.nbData = 100  # Number of datapoints
param.rfactor = 1e-9  # Control weight term

R = np.eye((param.nbData-1) * param.nbVarPos) * param.rfactor  # Control cost matrix

# Defin the viapoint passe time 
tl = np.linspace(0, param.nbData, param.nbPoints+1)
tl = np.rint(tl[1:]).astype(np.int64) - 1 

idx = np.array([i + np.arange(0,param.nbVar,1) for i in (tl*param.nbVar)]).flatten() 

# Viapoints
MuTmp = np.vstack((np.random.rand(param.nbVarPos, param.nbPoints) * 5, np.zeros(((param.nbVar-param.nbVarPos), param.nbPoints))))
param.Mu = MuTmp.reshape((param.nbPoints*param.nbVar,1), order='F')
print(MuTmp)
print(param.Mu.shape)
# Precision matrix
Qtmp = np.diag(np.hstack((np.ones(param.nbVarPos), np.zeros(param.nbVar-param.nbVarPos))));
Q = np.kron(np.eye(param.nbPoints), Qtmp)


# Dynamical System settings (discrete)
# =====================================
A1d = np.zeros((param.nbDeriv,param.nbDeriv))
B1d = np.zeros((param.nbDeriv,1))

for i in range(param.nbDeriv):
    A1d += np.diag(np.ones(param.nbDeriv-i), i) * param.dt**i 
    B1d[param.nbDeriv-i-1] = param.dt**(i+1) 


# A = np.eye(param.nbv)
A = np.kron(A1d,np.eye(param.nbVarPos))
B = np.kron(B1d,np.eye(param.nbVarPos))
print(A)
print(B)

# Build Sx and Su transfer matrices
Su0 = np.zeros((param.nbVar*param.nbData,param.nbVarPos * (param.nbData-1))) 
Sx0 = np.kron(np.ones((param.nbData,1)),np.eye(param.nbVar,param.nbVar))
M = B
for i in range(1,param.nbData):
    Sx0[i*param.nbVar:param.nbData*param.nbVar,:] = np.dot(Sx0[i*param.nbVar:param.nbData*param.nbVar,:], A)
    Su0[param.nbVar*i:param.nbVar*i+M.shape[0],0:M.shape[1]] = M
    M = np.hstack((np.dot(A,M),B))

Su = Su0[idx,:] # Keeping rows of interest
Sx = Sx0[idx,:] # Keeping rows of interest


# Batch LQR Reproduction
# =====================================
x0 = np.zeros((param.nbVar,1))
u_hat = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ (param.Mu - Sx @ x0)
x_hat = Sx0 @ x0 + Su0 @ u_hat


# Plotting
# =========
plt.figure()
plt.title("2D Trajectory")
plt.scatter(x_hat[0],x_hat[1],c='black',s=100)
plt.scatter(param.Mu[::param.nbVar],param.Mu[1::param.nbVar],c='red',s=100)  
plt.plot(x_hat[::param.nbVar] , x_hat[1::param.nbVar], c='black')
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

fig,axs = plt.subplots(2,1)
axs[0].scatter(tl,param.Mu[0::param.nbVar],c='red')  
axs[0].plot(x_hat[::param.nbVar], c='black')
axs[0].set_ylabel("$x_1$")
axs[0].set_xticks([0,param.nbData])
axs[0].set_xticklabels(["0","T"])

axs[1].scatter(tl,param.Mu[1::param.nbVar],c='red')  
axs[1].plot(x_hat[1::param.nbVar], c='black')
axs[1].set_ylabel("$x_2$")
axs[1].set_xlabel("$t$")
axs[1].set_xticks([0,param.nbData])
axs[1].set_xticklabels(["0","T"])

plt.show()
