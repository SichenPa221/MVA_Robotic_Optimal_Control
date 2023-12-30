'''
iLQR applied to a planar bimanual robot problem with a cost on tracking a desired
manipulability ellipsoid at the center of mass (batch formulation)

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Boyang Ti <https://tiboy.co/> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.linalg
from scipy.linalg import fractional_matrix_power


# Helper functions
# ===============================

# Forward kinematics for end-effector (in robot coordinate system)
def fkin(x, param):
    L = np.tril(np.ones(3))
    f = np.vstack([
        param.l[0:3].T @ np.cos(L @ x[0:3]),
        param.l[0:3].T @ np.sin(L @ x[0:3]),
        param.l[[0,3,4]].T @ np.cos(L @ x[[0,3,4]]),
        param.l[[0,3,4]].T @ np.sin(L @ x[[0,3,4]])
    ])  # f1,f2,f3,f4
    return f

# Forward kinematics for end-effector (in robot coordinate system)
def fkin0(x, param): 
    L = np.tril(np.ones(3))
    fl = np.vstack([
        L @ np.diag(param.l[0:3]) @ np.cos(L @ x[0:3]),
        L @ np.diag(param.l[0:3]) @ np.sin(L @ x[0:3])
    ])
    fr = np.vstack([
        L @ np.diag(param.l[[0,3,4]]) @ np.cos(L @ x[[0,3,4]]),
        L @ np.diag(param.l[[0,3,4]]) @ np.sin(L @ x[[0,3,4]])
    ])
    f = np.hstack([fl[:,::-1], np.zeros([2,1]), fr])
    return f

# Jacobian of the end-effector with analytical computation (for single time step)
def Jkin(x, param):
    L = np.tril(np.ones(3))
    J = np.zeros((param.nbVarF, param.nbVarX))
    Jl = np.vstack([-np.sin(L @ x[:3]).T @ np.diag(param.l[:3]) @ L,
                    np.cos(L @ x[:3]).T @ np.diag(param.l[:3]) @ L
                    ])
    Jr = np.vstack([-np.sin(L @ x[[0,3,4]]).T @ np.diag(np.array(param.l)[[0,3,4]]) @ L,
                    np.cos(L @ x[[0,3,4]]).T @ np.diag(np.array(param.l)[[0,3,4]]) @ L
                    ])
    J[:Jl.shape[0], :Jl.shape[1]] = Jl
    J[2:, [0,3,4]] = Jr
    return J

# Forward kinematics for center of mass of a bimanual robot (in robot coordinate system)
def fkin_CoM(x, param):
    L = np.tril(np.ones(3))
    f = np.vstack((param.l[0:3].T @ L @ np.cos(L @ x[0:3]) + param.l[[0,3,4]].T @ L @ np.cos(L @ x[[0,3,4]]),
                   param.l[0:3].T @ L @ np.sin(L @ x[0:3]) + param.l[[0,3,4]].T @ L @ np.sin(L @ x[[0,3,4]]))) / 6

    return f

# Jacobian  of the center of mass with analytical computation (for single time step)
def Jkin_CoM(x, param):
    L = np.tril(np.ones(3))
    Jl = np.vstack([-np.sin(L @ x[:3]).T @ L @ np.diag(param.l[:3].T @ L),
                    np.cos(L @ x[:3]).T @ L @ np.diag(param.l[:3].T @ L)
                    ]) / 6
    Jr = np.vstack([-np.sin(L @ x[[0,3,4]]).T @ L @ np.diag(np.array(param.l)[[0,3,4]].T @ L),
                    np.cos(L @ x[[0,3,4]]).T @ L @ np.diag(np.array(param.l)[[0,3,4]].T @ L)
                    ]) / 6
    #J = np.hstack(((Jl[:,0] + Jr[:,0]).reshape(-1,1), Jl[:,1:], Jr[:,1:]))
    J = np.hstack([(Jl[:,0] + Jr[:,0])[:,np.newaxis], Jl[:,1:], Jr[:,1:]])
    return J

def rman(x, param):
    G = fractional_matrix_power(param.MuS, -0.5)
    f = np.zeros((3, np.size(x,1)))
    for i in range(np.size(x,1)):
        Jt = Jkin_CoM(x[:,i], param)  # Jacobian for center of mass
        St = Jt @ Jt.T  # manipulability matrix

        D, V = np.linalg.eig(G @ St @ G)
        E = V @ np.diag(np.log(D)) @ np.linalg.pinv(V)

        E = np.tril(E) * (np.eye(2) + np.tril(np.ones(2), -1) * np.sqrt(2))
        f[:,i] = E[np.where(E!=0)]
    return f

# Jacobian for manipulability tracking with numerical computation
def Jman_num(x, param):
    e = 1E-6
    X = np.matlib.repmat(x, 1, param.nbVarX)
    F1 = rman(X, param)
    F2 = rman(X + np.eye(param.nbVarX) * e, param)
    J = (F2 - F1) / e
    return J

# Residuals f and Jacobians J for manipulability tracking
# (c=f'*f is the cost, g=J'*f is the gradient, H=J'*J is the approximated Hessian)
def f_manipulability(x, param):
    f = rman(x, param)  # Residuals
    for t in range(np.size(x, 1)):
        if t == 0:
            J = Jman_num(x[:,t].reshape(-1, 1), param)
        else:
            J = scipy.linalg.block_diag(J, Jman_num(x[:,t].reshape(-1, 1), param))  # Jacobians
    return f, J

## Parameters
param = lambda: None # Lazy way to define an empty class in python

param.dt = 1e0 # Time step length
param.nbIter = 100 # Maximum number of iterations for iLQR
param.nbPoints = 1 # Number of viapoints
param.nbData = 10 # Number of datapoints
param.nbVarX = 5 # State space dimension ([q1,q2,q3] for left arm, [q1,q4,q5] for right arm)
param.nbVarU = param.nbVarX # Control space dimension ([dq1, dq2, dq3, dq4, dq5])
param.nbVarF = 4 # Objective function dimension ([x1,x2] for left arm and [x3,x4] for right arm)
param.l = np.ones(param.nbVarX) * 2 # Robot links lengths
param.r = 1e-6 # Control weight term
param.MuS = np.array([[10,2], [2,4]])

# Precision matrix
Q = np.kron(np.identity(param.nbPoints), np.diag([0., 0., 0., 0.]))
# Control weight matrix
R = np.identity((param.nbData-1) * param.nbVarU) * param.r
# Precision matrix for continuous CoM tracking
Qc = np.kron(np.identity(param.nbData), np.diag([0., 0.]))

# Time occurrence of viapoints
tl = np.linspace(0, param.nbData, param.nbPoints+1)
tl = np.rint(tl[1:]).astype(np.int64) - 1
idx = np.array([i + np.arange(0, param.nbVarX, 1) for i in (tl*param.nbVarX)])

# iLQR
# ===============================
u = np.zeros(param.nbVarU * (param.nbData-1))  # Initial control command
x0 = np.array([np.pi/3, np.pi/2, np.pi/3, -np.pi/3, -np.pi/4])  # Initial state

# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([
    np.zeros([param.nbVarX, param.nbVarX*(param.nbData-1)]),
    np.tril(np.kron(np.ones([param.nbData-1, param.nbData-1]), np.eye(param.nbVarX) * param.dt))
])
Sx0 = np.kron(np.ones(param.nbData), np.identity(param.nbVarX)).T
Su = Su0[idx.flatten()]  # We remove the lines that are out of interest
for i in range(param.nbIter):
    x = Su0 @ u + Sx0 @ x0  # System evolution
    x = x.reshape([param.nbVarX, param.nbData], order='F')
    f, J = f_manipulability(x[:,tl], param)  # Residuals and Jacobians
    du = np.linalg.inv(Su.T @ J.T @ J @ Su + R) @ (-Su.T @ J.T @ f.flatten('F') - u * param.r)  # Gauss-Newton update
    # Estimate step size with backtracking line search method
    alpha = 1
    cost0 = f.flatten('F').T @ f.flatten('F') + np.linalg.norm(u)**2 * param.r  # Cost
    while True:
        utmp = u + du * alpha
        xtmp = Su0 @ utmp + Sx0 @ x0  # System evolution
        xtmp = xtmp.reshape([param.nbVarX, param.nbData], order='F')
        ftmp, _ = f_manipulability(xtmp[:, tl], param)  # Residuals
        cost = ftmp.flatten('F').T @ ftmp.flatten('F') + np.linalg.norm(utmp)**2 * param.r  # Cost
        if cost < cost0 or alpha < 1e-3:
            u = utmp
            print("Iteration {}, cost: {}".format(i,cost))
            break
        alpha /= 2
    if np.linalg.norm(du * alpha) < 1E-2:
        break # Stop iLQR iterations when solution is reached

# Plots
# ===============================
plt.figure()
plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')

fc = fkin_CoM(x, param)
al = np.linspace(-np.pi, np.pi, 50)
ax = plt.gca()

# Plot desired manipulability ellipsoid
D1, V1 = np.linalg.eig(param.MuS)
D1 = np.diag(D1)
R1 = np.real(V1 @ np.sqrt(D1+0j))
msh1 = (R1 @ np.array([np.cos(al), np.sin(al)]) * 0.52).T + np.matlib.repmat(fc[:, -1].reshape(-1, 1), 1, 50).T
p1 = patches.Polygon(msh1, closed=True, alpha=0.9)
p1.set_facecolor([1, 0.7, 0.7])
p1.set_edgecolor([1, 0.6, 0.6])
ax.add_patch(p1)

# Plot robot manipulability ellipsoid
J = Jkin_CoM(x[:, -1], param)
S = J @ J.T
D2, V2 = np.linalg.eig(S)
D2 = np.diag(D2)
R2 = np.real(V2 @ np.sqrt(D2+0j))
msh2 = (R2 @ np.array([np.cos(al), np.sin(al)]) * 0.5).T + np.matlib.repmat(fc[:, -1].reshape(-1, 1), 1, 50).T
p2 = patches.Polygon(msh2, closed=True, alpha=0.9)
p2.set_facecolor([0.4, 0.4, 0.4])
p2.set_edgecolor([0.3, 0.3, 0.3])
ax.add_patch(p2)

# Plot CoM
fc = fkin_CoM(x, param) # Forward kinematics for center of mass
plt.plot(fc[0,0], fc[1,0], marker='o', markerfacecolor='none', markeredgewidth=4, markersize=10, markeredgecolor=[0.5, 0.5, 0.5]) # Plot CoM
plt.plot(fc[0,tl[-1]], fc[1,tl[-1]], marker='o', markerfacecolor='none', markeredgewidth=4, markersize=10, markeredgecolor=[0.2, 0.2, 0.2]) # Plot CoM

# Plot end-effectors paths
f01 = fkin0(x[:,0], param)
f02 = fkin0(x[:,tl[0]], param)

# Get points of interest
f = fkin(x, param)
plt.plot(f01[0,:], f01[1,:], c='black', linewidth=4, alpha=.2)
plt.plot(f02[0,:], f02[1,:], c='black', linewidth=4, alpha=.4)

plt.plot(f[0,:], f[1,:], c='black', marker='o', markevery=[0]+tl.tolist())
plt.plot(f[2,:], f[3,:], c='black', marker='o', markevery=[0]+tl.tolist())

plt.show()
