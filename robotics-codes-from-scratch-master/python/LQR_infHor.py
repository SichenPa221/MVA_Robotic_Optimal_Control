'''
Point-mass LQR with infinite horizon

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Jérémy Maceiras <jeremy.maceiras@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import copy
from scipy.linalg import solve_discrete_are as solve_algebraic_riccati_discrete
from scipy.stats import multivariate_normal
from math import factorial
import matplotlib.pyplot as plt

# Plot a 2D Gaussians
def plot_gaussian(mu, sigma):
    w, h = 100, 100

    std = [np.sqrt(sigma[0, 0]), np.sqrt(sigma[1, 1])]
    x = np.linspace(mu[0] - 3 * std[0], mu[0] + 3 * std[0], w)
    y = np.linspace(mu[1] - 3 * std[1], mu[1] + 3 * std[1], h)

    x, y = np.meshgrid(x, y)

    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T

    normal_rv = multivariate_normal(mu, sigma)
    z = normal_rv.pdf(xy)
    z = z.reshape(w, h, order='F')

    plt.contourf(x, y, z.T,levels=1,colors=["white","red"],alpha=.5)

# Parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.nbData = 200 # Number of datapoints
param.nbRepros = 4 #Number of reproductions
param.nbVarPos = 2 #Dimension of position data (here: x1,x2)
param.nbDeriv = 2 #Number of static & dynamic features (D=2 for [x,dx])
param.nbVar = param.nbVarPos * param.nbDeriv #Dimension of state vector in the tangent space
param.dt = 1e-2 #Time step duration
param.rfactor = 4e-2	#Control cost in LQR 

# Control cost matrix
R = np.identity(param.nbVarPos) * param.rfactor

# Target and desired covariance
param.Mu = np.hstack(( np.random.uniform(size=param.nbVarPos) , np.zeros(param.nbVarPos) ))
Ar,_ = np.linalg.qr(np.random.uniform(size=(param.nbVarPos,param.nbVarPos)))
xCov = Ar @ np.diag(np.random.uniform(size=param.nbVarPos)) @ Ar.T * 1e-1

# Discrete dynamical System settings 
# ===================================

A1d = np.zeros((param.nbDeriv,param.nbDeriv))
B1d = np.zeros((param.nbDeriv,1))

for i in range(param.nbDeriv):
    A1d += np.diag( np.ones(param.nbDeriv-i) ,i ) * param.dt**i * 1/factorial(i)
    B1d[param.nbDeriv-i-1] = param.dt**(i+1) * 1/factorial(i+1)

A = np.kron(A1d,np.identity(param.nbVarPos))
B = np.kron(B1d,np.identity(param.nbVarPos))

# Discrete LQR with infinite horizon
# ===================================

Q = np.zeros((param.nbVar,param.nbVar))
Q[:param.nbVarPos,:param.nbVarPos] = np.linalg.inv(xCov) # Precision matrix
P = solve_algebraic_riccati_discrete(A,B,Q,R)
L = np.linalg.inv(B.T @ P @ B + R) @ B.T @ P @ A # Feedback gain (discrete version)

reproducitons = []

for i in range(param.nbRepros):
    xt = np.zeros(param.nbVar)
    xt[:param.nbVarPos] = 1+np.random.uniform(param.nbVarPos)*2 
    xs = [copy.deepcopy(xt)]
    for t in range(param.nbData):
        u = L @ (param.Mu - xt)
        xt = A @ xt + B @ u
        xs += [copy.deepcopy(xt)]
    reproducitons += [ np.asarray(xs) ]

# Plots
# ======

plt.figure()
plt.title("Position")

for r in reproducitons:
    plt.plot(r[:,0],r[:,1],c="black")
plot_gaussian(param.Mu[:param.nbVarPos],xCov)
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

fig,axs = plt.subplots(5)

for r in reproducitons:
    axs[0].plot(r[:,0],c="black",alpha=.4,linestyle="dashed")
    axs[1].plot(r[:,1],c="black",alpha=.4,linestyle="dashed")    
    axs[2].plot(r[:,2],c="black",alpha=.4,linestyle="dashed")
    axs[3].plot(r[:,3],c="black",alpha=.4,linestyle="dashed")
    axs[4].plot(np.linalg.norm(r[:,2:4],axis=1),c="black",alpha=.4,linestyle="dashed")

axs[0].set_ylabel("$x_1$")
axs[1].set_ylabel("$x_2$")
axs[2].set_ylabel("$\dot{x}_1$")
axs[3].set_ylabel("$\dot{x}_2$")
axs[4].set_ylabel("$| \dot{x} |$")
plt.xlabel("T")

plt.show()
