'''
Batch LQT with nullspace formulation

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Hakan Girgin <hakan.girgin@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

## Precomputation of basis functions to generate structured stochastic u through Bezier curves
# Building Bernstein basis functions
def build_phi_bernstein(nb_data, nb_fct):
    t = np.linspace(0,1,nb_data)
    phi = np.zeros((nb_data,nb_fct))
    for i in range(nb_fct):
        phi[:,i] = factorial(nb_fct-1) / (factorial(i) * factorial(nb_fct-1-i)) * (1-t)**(nb_fct-1-i) * t**i
    return phi

# Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-1  # Time step length
param.nbPoints = 1 # Number of targets
param.nbDeriv = 1 # Order of the dynamical system
param.nbVarPos = 2 # Number of position variable
param.nbVar = param.nbVarPos * param.nbDeriv # Dimension of state vector
param.nbData = 50  # Number of datapoints
param.rfactor = 1e-4  # Control weight term
param.nbRepros = 60 # Number of stochastic reproductions


# Dynamical System settings (discrete)
# =====================================
A1d = np.zeros((param.nbDeriv,param.nbDeriv))
B1d = np.zeros((param.nbDeriv,1))

for i in range(param.nbDeriv):
    A1d += np.diag(np.ones(param.nbDeriv-i), i) * param.dt**i * 1/factorial(i)
    B1d[param.nbDeriv-i-1] = param.dt**(i+1) * 1/factorial(i+1)

A = np.kron(A1d, np.eye(param.nbVarPos))
B = np.kron(B1d, np.eye(param.nbVarPos))

# Build Sx and Su transfer matrices
Su = np.zeros((param.nbVar*param.nbData,param.nbVarPos * (param.nbData-1)))
Sx = np.kron(np.ones((param.nbData,1)), np.eye(param.nbVar,param.nbVar))

M = B
for i in range(1, param.nbData):
    Sx[i*param.nbVar:param.nbData*param.nbVar,:] = np.dot(Sx[i*param.nbVar:param.nbData*param.nbVar,:], A)
    Su[param.nbVar*i:param.nbVar*i+M.shape[0],0:M.shape[1]] = M
    M = np.hstack((np.dot(A,M),B)) # [0,nb_state_var-1]

# Cost function settings
# =====================================
R = np.eye((param.nbData-1) * param.nbVarPos) * param.rfactor  # Control cost matrix

t_list = np.stack([param.nbData-1]) # viapoint time list
mu_list = np.stack([np.array([20,10.])]) # viapoint list
Q_list = np.stack([np.eye(param.nbVar)*1e3]) # viapoint precision list

mus = np.zeros((param.nbData, param.nbVar))
Qs = np.zeros((param.nbData, param.nbVar, param.nbVar))
for i,t in enumerate(t_list):
    mus[t] = mu_list[i]
    Qs[t] = Q_list[i]
muQ = mus.flatten()
Q  = block_diag(*Qs)

# Change here off block diagonals of Q


# Batch LQR Reproduction
# =====================================
nbRBF = 10
H = build_phi_bernstein(param.nbData-1, nbRBF)

J = Q @ Su
N = np.eye(J.shape[-1]) - np.linalg.pinv(J) @ J # nullspace operator of LQT

# Principal Task
x0 = np.zeros(param.nbVar)
u1 = np.linalg.solve(Su.T @ Q @ Su + R,  Su.T @ Q @ (muQ - Sx @ x0))
x1 = Sx @ x0 + Su @ u1

# Secondary Task
repr_x = np.zeros((param.nbRepros, param.nbData*param.nbVar))
for n in range(param.nbRepros):
    w = np.random.randn(nbRBF, param.nbVarPos) * 1E1 # Random weights
    u2 = H @ w # Reconstruction of control signals by a weighted superposition of basis functions
    u = u1 + N @ u2.flatten()
    repr_x[n] = Sx @ x0 + Su @ u

# Plotting
# =========
plt.figure()
plt.title("2D Trajectory")
plt.scatter(x1[0],x1[1],c='black',s=100)
for mu in mu_list:
    plt.scatter(mu[0], mu[1],c='red',s=100)

plt.plot(x1[::param.nbVar] , x1[1::param.nbVar], c='black')
for i in range(param.nbRepros):
    plt.plot(repr_x[i,::param.nbVar], repr_x[i,1::param.nbVar], c='blue', alpha=0.1, zorder=0)

plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

fig,axs = plt.subplots(2,1)
for i,t in enumerate(t_list):
    axs[0].scatter(t, mu_list[i][0],c='red')

for i in range(param.nbRepros):
    axs[0].plot(repr_x[i,::param.nbVar], c='blue', alpha=0.1, zorder=0)

axs[0].plot(x1[::param.nbVar], c='black')
axs[0].set_ylabel("$x_1$")
axs[0].set_xticks([0,param.nbData])
axs[0].set_xticklabels(["0","T"])

for i,t in enumerate(t_list):
    axs[1].scatter(t, mu_list[i][1],c='red')

for i in range(param.nbRepros):
    axs[1].plot(repr_x[i,1::param.nbVar], c='blue', alpha=0.1, zorder=0)

axs[1].plot(x1[1::param.nbVar], c='black')
axs[1].set_ylabel("$x_2$")
axs[1].set_xlabel("$t$")
axs[1].set_xticks([0,param.nbData])
axs[1].set_xticklabels(["0","T"])

plt.show()
