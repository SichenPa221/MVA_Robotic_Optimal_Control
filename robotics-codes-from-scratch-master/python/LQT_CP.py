'''
Linear quadratic tracking (LQT) with control primitives applied to a viapoints example

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://rcfs.ch/>
License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from math import factorial

# Building piecewise constant basis functions
def build_phi_piecewise(nb_data, nb_fct):
    phi = np.kron( np.identity(nb_fct) , np.ones((int(np.ceil(nb_data/nb_fct)),1)) )
    return phi[:nb_data]

# Building radial basis functions (RBFs)
def build_phi_rbf(nb_data, nb_fct):
    t = np.linspace(0,1,nb_data).reshape((-1,1))
    tMu = np.linspace( t[0] , t[-1] , nb_fct )
    phi = np.exp( -1e2 * (t.T - tMu)**2 )
    return phi.T

# Building Bernstein basis functions
def build_phi_bernstein(nb_data, nb_fct):
    t = np.linspace(0,1,nb_data)
    phi = np.zeros((nb_data,nb_fct))
    for i in range(nb_fct):
        phi[:,i] = factorial(nb_fct-1) / (factorial(i) * factorial(nb_fct-1-i)) * (1-t)**(nb_fct-1-i) * t**i
    return phi

# Building Fourier basis functions
def build_phi_fourier(nb_data, nb_fct):

    t = np.linspace(0,1,nb_data).reshape((-1,1))

    # Alternative computation for real and even signal
    k = np.arange(0,nb_fct).reshape((-1,1))
    phi = np.cos( t.T * k * 2 * np.pi )
    return phi.T

# General parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-2  # Time step length
param.nbPoints = 3 # Number of target points
param.nbDeriv = 2 # Order of the dynamics model
param.nbVarPos = 2 # Number of position variable
param.nbVar = param.nbVarPos * param.nbDeriv # Dimension of state vector
param.nbData = 100  # Number of datapoints
param.nbFct = 3  # Number of basis function
param.rfactor = 1e-9  # Control weight term
param.basisName = "RBF"  # can be PIECEWISE, RBF, BERNSTEIN, FOURIER

R = np.eye((param.nbData-1) * param.nbVarPos) * param.rfactor  # Control cost matrix

tl = np.linspace(0,param.nbData,param.nbPoints+1)
tl = np.rint(tl[1:]).astype(np.int64)-1 
idx = np.array([i + np.arange(0,param.nbVar,1) for i in (tl*param.nbVar)]).flatten() 

# Viapoints
MuTmp = np.vstack((np.random.rand(param.nbVarPos, param.nbPoints) * 5, np.zeros(((param.nbVar-param.nbVarPos), param.nbPoints))))
param.Mu = MuTmp.reshape((param.nbPoints*param.nbVar,1), order='F')

# Precision matrix
Qtmp = np.diag(np.hstack((np.ones(param.nbVarPos), np.zeros(param.nbVar-param.nbVarPos))));
Q = np.kron(np.eye(param.nbPoints), Qtmp)


# Dynamical System settings (discrete)
# =====================================
A = np.identity(param.nbVar)
if param.nbDeriv==2:
    A[:param.nbVarPos,-param.nbVarPos:] = np.identity(param.nbVarPos) * param.dt

B = np.zeros(( param.nbVar , param.nbVarPos ))
derivatives = [ param.dt,param.dt**2 /2 ][:param.nbDeriv]
for i in range(param.nbDeriv):
    B[i*param.nbVarPos:(i+1)*param.nbVarPos] = np.identity(param.nbVarPos) * derivatives[::-1][i]

# Build Sx and Su transfer matrices
Su0 = np.zeros((param.nbVar*param.nbData,param.nbVarPos * (param.nbData-1))) # It's maybe n-1 not sure
Sx0 = np.kron(np.ones((param.nbData,1)),np.eye(param.nbVar,param.nbVar))
M = B
for i in range(1,param.nbData):
    Sx0[i*param.nbVar:param.nbData*param.nbVar,:] = np.dot(Sx0[i*param.nbVar:param.nbData*param.nbVar,:],A)
    Su0[param.nbVar*i:param.nbVar*i+M.shape[0],0:M.shape[1]] = M
    M = np.hstack((np.dot(A,M),B)) # [0,nb_state_var-1]

Su = Su0[idx,:] # Keeping rows of interest
Sx = Sx0[idx,:] # Keeping rows of interest


# Building basis functions
# =========================
functions = {
    "PIECEWISE": build_phi_piecewise ,
    "RBF": build_phi_rbf,
    "BERNSTEIN": build_phi_bernstein,
    "FOURIER": build_phi_fourier
}
phi = functions[param.basisName](param.nbData-1,param.nbFct)
PSI = np.kron(phi,np.identity(param.nbVarPos))

# Batch LQR Reproduction
# =====================================
x0 = np.zeros((param.nbVar,1))
w_hat = np.linalg.inv(PSI.T @ Su.T @ Q @ Su @ PSI + PSI.T @ R @ PSI) @ PSI.T @ Su.T @ Q @ (param.Mu - Sx @ x0)
u_hat = PSI @ w_hat
x_hat = Sx0 @ x0 + Su0 @ u_hat

# Plotting
# =========
plt.figure()

plt.title("2D Trajectory")
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(x_hat[0],x_hat[1],c='black',s=100)
plt.scatter(param.Mu[::param.nbVar],param.Mu[1::param.nbVar],c='red',s=100)
plt.plot(x_hat[::param.nbVar], x_hat[1::param.nbVar], c='black')

fig,axs = plt.subplots(5,1)

axs[0].scatter(tl,param.Mu[0::param.nbVar],c='red')  
axs[0].plot(x_hat[::param.nbVar],c='black')
axs[0].set_ylabel("$x_1$")
axs[0].set_xticks([0,param.nbData])
axs[0].set_xticklabels(["0","T"])

axs[1].scatter(tl,param.Mu[1::param.nbVar],c='red')  
axs[1].plot(x_hat[1::param.nbVar],c='black')
axs[1].set_ylabel("$x_2$")
axs[1].set_xticks([0,param.nbData])
axs[1].set_xticklabels(["0","T"])

axs[2].plot(u_hat[::param.nbVarPos],c='black')
axs[2].set_ylabel("$u_1$")
axs[2].set_xticks([0,param.nbData-1])
axs[2].set_xticklabels(["0","T-1"])

axs[3].plot(u_hat[1::param.nbVarPos],c='black')
axs[3].set_ylabel("$u_2$")
axs[3].set_xticks([0,param.nbData-1])
axs[3].set_xticklabels(["0","T-1"])

axs[4].set_ylabel("$\phi_k$")
axs[4].set_xticks([0,param.nbData-1])
axs[4].set_xticklabels(["0","T-1"])
for i in range(param.nbFct):
    axs[4].plot(phi[:,i])
axs[4].set_xlabel("$t$")

plt.show()
