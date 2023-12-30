'''
Linear Quadratic tracker applied on a via point example while coordinating two agents

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Julius Jankowski <julius.jankowski@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
from math import factorial
import matplotlib.pyplot as plt

# Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-2  # Time step length
param.nbPoints = 1 # Number of target points
param.nbDeriv = 2 # Order of the dynamical system
param.nbVarPos = 2 # Number of position variable
param.nbVar = 2 * param.nbVarPos * param.nbDeriv + 1 # Dimension of state vector: [x_1, x_2, dx_1, dx_2, 1]
param.nbVarU = 2 * param.nbVarPos
param.nbData = 100  # Number of datapoints
param.rfactor = 1e-9  # Control weight term

R = np.eye((param.nbData-1) * param.nbVarU) * param.rfactor  # Control cost matrix
Q = np.zeros((param.nbVar * param.nbData, param.nbVar * param.nbData)) # Task precision for augmented state

via_points = []
via_point_timing = np.linspace(0, param.nbData-1, param.nbPoints+1)

for tv_ in via_point_timing[1:]:
    tv = int(tv_)
    tv_slice = slice(tv*param.nbVar, (tv+1)*param.nbVar)
    
    X_d_tv = np.concatenate((np.random.uniform(-1.0, 1.0, size=2 * param.nbVarPos), np.zeros(2 * param.nbVarPos)))
    
    via_points.append(X_d_tv[:2*param.nbVarPos])
    print("via point: ", X_d_tv)
    
    Q_tv = np.eye(param.nbVar-1)
    if tv < param.nbData-1:
      Q_tv[2*param.nbVarPos:, 2*param.nbVarPos:] = 1e-6*np.eye(2*param.nbVarPos) # Don't track velocities on the way
    
    Q_inv = np.zeros((param.nbVar, param.nbVar))
    Q_inv[:param.nbVar-1, :param.nbVar-1] = np.linalg.inv(Q_tv) + np.outer(X_d_tv, X_d_tv)
    Q_inv[:param.nbVar-1, -1] = X_d_tv
    Q_inv[-1, :param.nbVar-1] = X_d_tv
    Q_inv[-1, -1] = 1
    
    Q[tv_slice, tv_slice] = np.linalg.inv(Q_inv)
    
# Construct meeting point constraint at half of the horizon
tv = int(0.5 * param.nbData)
tv_slice = slice(tv*param.nbVar, (tv+1)*param.nbVar)
Q_tv = np.eye(param.nbVar-1)

# Off-diagonal to tie position
Q_tv[:param.nbVarPos, param.nbVarPos:2*param.nbVarPos] = - (1.0 - 1e-6) * np.eye(param.nbVarPos)
Q_tv[param.nbVarPos:2*param.nbVarPos, :param.nbVarPos] = - (1.0 - 1e-6) * np.eye(param.nbVarPos)

# Off-diagonal to tie velocity
Q_tv[2*param.nbVarPos:3*param.nbVarPos, 3*param.nbVarPos:4*param.nbVarPos] = - (1.0 - 1e-6) * np.eye(param.nbVarPos)
Q_tv[3*param.nbVarPos:4*param.nbVarPos, 2*param.nbVarPos:3*param.nbVarPos] = - (1.0 - 1e-6) * np.eye(param.nbVarPos)

#Q_tv[2*param.nbVarPos:, 2*param.nbVarPos:] = 1e-6*np.eye(2*param.nbVarPos) # Don't track absolute velocities
Q_inv = np.zeros((param.nbVar, param.nbVar))
Q_inv[:param.nbVar-1, :param.nbVar-1] = np.linalg.inv(Q_tv)
Q_inv[-1, -1] = 1
Q[tv_slice, tv_slice] = np.linalg.inv(Q_inv)


# Dynamical System settings (discrete)
# =====================================
A1d = np.zeros((param.nbDeriv,param.nbDeriv))
B1d = np.zeros((param.nbDeriv,1))

for i in range(param.nbDeriv):
    A1d += np.diag( np.ones(param.nbDeriv-i), i ) * param.dt**i * 1/factorial(i)
    B1d[param.nbDeriv-i-1] = param.dt**(i+1) * 1/factorial(i+1)

A = np.eye(param.nbVar)
A[:param.nbVar-1, :param.nbVar-1] = np.kron(A1d, np.identity(2 * param.nbVarPos))
B = np.zeros((param.nbVar, 2 * param.nbVarPos))
B[:param.nbVar-1] = np.kron(B1d, np.identity(2 * param.nbVarPos))

# Build Sx and Su transfer matrices
Su = np.zeros((param.nbVar*param.nbData, param.nbVarU * (param.nbData-1))) 
Sx = np.kron(np.ones((param.nbData,1)), np.eye(param.nbVar,param.nbVar))

M = B
for i in range(1,param.nbData):
    Sx[i*param.nbVar:param.nbData*param.nbVar,:] = np.dot(Sx[i*param.nbVar:param.nbData*param.nbVar,:], A)
    Su[param.nbVar*i:param.nbVar*i+M.shape[0],0:M.shape[1]] = M
    M = np.hstack((np.dot(A,M),B)) # [0,nb_state_var-1]


# Build recursive least squares solution for the feedback gains, using u_k = - F_k @ x0 = - K_k @ x_k
# =====================================
F = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ Sx
K = np.zeros((param.nbData-1, param.nbVarU, param.nbVar))
P = np.eye(param.nbVar)
K[0] = F[:param.nbVarU]
for k in range(1, param.nbData-1):
  k_slice = slice(k*param.nbVarU, (k+1)*param.nbVarU)
  P = P @ np.linalg.inv(A - B @ K[k-1])
  K[k] = F[k_slice] @ P


# Multiple reproductions from different initial positions with perturbation
# =====================================
num_repro = 1

x = np.zeros((param.nbVar, num_repro))
x[:param.nbVarPos, :] = np.random.normal(-0.5, 1e-1, (param.nbVarPos, num_repro))
x[param.nbVarPos:2*param.nbVarPos, :] = np.random.normal(0.5, 1e-1, (param.nbVarPos, num_repro))
x[-1,:] = np.ones(num_repro)

X_batch = []
X_batch.append(x[:-1,:].copy())

for k in range(param.nbData-1):
  u = - K[k] @ x
  x = A @ x + B @ (u + np.random.normal(0, 5e+0, (param.nbVarU, num_repro)))
  X_batch.append(x[:-1,:].copy())
  
X_traj = np.array(X_batch)


# Plotting
# =========
plt.figure()
plt.title("2D Trajectory")
# Agent 1
plt.scatter(X_traj[0,0], X_traj[0,1], c='black', s=100)
plt.plot(X_traj[:,0], X_traj[:,1], c='black')
for p in via_points:
  plt.scatter(p[0], p[1], c='red', s=100)
  
# Agent 2
plt.scatter(X_traj[0,2], X_traj[0,3], c='black', s=100)
plt.plot(X_traj[:,2], X_traj[:,3], c='black')
for p in via_points:
  plt.scatter(p[2], p[3], c='red', s=100)  

plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

plt.show()
