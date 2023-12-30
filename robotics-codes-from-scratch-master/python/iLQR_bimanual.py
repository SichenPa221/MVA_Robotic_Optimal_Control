'''
iLQR applied to a planar bimanual robot for a tracking problem involving
the center of mass (CoM) and the end-effector (batch formulation)

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
#import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

# Residual and Jacobian of end-effector for a viapoints reaching task (in robot coordinate system)
def f_reach(x, param):
    f = fkin(x, param) - param.Mu
    J = np.zeros((param.nbVarF*param.nbPoints, param.nbVarX*param.nbPoints))
    for t in range(param.nbPoints):
        Jtmp = Jkin(x[:,t], param)
        J[t * param.nbVarF:(t + 1) * param.nbVarF, t * param.nbVarX:(t + 1) * param.nbVarX] = Jtmp
    return f, J
        
# Forward kinematics of the center of mass
def fkin_CoM(x, param):
    L = np.tril(np.ones(3))
    f = np.vstack([param.l[:3] @ L @ np.cos(L @ x[:3,:]) +
                   np.array(param.l)[[0,3,4]] @ L @ np.cos(L @ x[[0,3,4],:]),
                   param.l[:3] @ L @ np.sin(L @ x[:3,:]) +
                   np.array(param.l)[[0,3,4]] @ L @ np.sin(L @ x[[0,3,4],:])
                   ]) / 6
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

# Residual and Jacobian of Center of Mass for a viapoints reaching task (in robot coordinate system)
def f_reach_CoM(x, param):
    #f = fkin_CoM(x, param) -np.matlib.repmat(param.MuCoM, 1, param.nbData)
    f = fkin_CoM(x, param) - param.MuCoM
    J = np.zeros([2*param.nbData, param.nbVarX*param.nbData])
    for t in range(param.nbData):
        Jtmp = Jkin_CoM(x[:,t], param)
        J[t*2:(t+1)*2, t*param.nbVarX:(t+1)*param.nbVarX] = Jtmp
    return f, J


param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e0 # Time step length
param.nbData = 30 # Number of datapoints
param.nbIter = 100 # Maximum number of iterations for iLQR
param.nbPoints = 1 # Number of viapoints
param.nbVarX = 5 # State space dimension ([q1,q2,q3] for left arm, [q1,q4,q5] for right arm)
param.nbVarU = param.nbVarX # Control space dimension (dq1,dq2,dq3,dq4,dq5)
param.nbVarF = 4 # Task space dimension ([x1,x2] for left end-effector, [x3,x4] for right end-effector)
param.l = np.ones(param.nbVarX) * 2 # Robot links lengths
    
param.r = 1e-5 # Control weight term
param.Mu = np.array([[-1, -1.5, 4, 2]]).T # Target point for end-effectors
param.MuCoM = np.array([[0, 1.4]]).T # Target point for center of mass

# Main program
# ===============================

# Control weight matrix (at trajectory level)
R = np.eye(param.nbVarU * (param.nbData-1)) * param.r
# Precision matrix for end-effectors tracking
Q = np.kron(np.eye(param.nbPoints), np.diag([1, 1, 0, 0]))
# Precision matrix for continuous CoM tracking
Qc = np.kron(np.eye(param.nbData), np.diag([1, 1]))

# Time occurence of viapoints
tl = np.linspace(0, param.nbData-1, param.nbPoints+1 )
tl = np.fromiter(map(lambda x: int(x + 0.5), tl[1:]), dtype=np.int32)
idx = (tl)[:,np.newaxis] * param.nbVarX + np.arange(param.nbVarU)

# initial setup
u = np.zeros(param.nbVarU * (param.nbData-1)) # Initial control command
x0 = np.array([np.pi/2, np.pi/2, np.pi/3, -np.pi/2, -np.pi/3])#Initial pose

# Solving Iterative LQR (iLQR)
# ===============================

# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([np.zeros((param.nbVarX, param.nbVarX*(param.nbData-1))),
      np.tril(np.kron(np.ones((param.nbData-1, param.nbData-1)), np.eye(param.nbVarX)*param.dt))])
Sx0 = np.kron( np.ones(param.nbData) , np.identity(param.nbVarX) ).T
Su = Su0[idx.flatten()] # We remove the lines that are out of interest

for i in range(param.nbIter):
    x = Su0 @ u + Sx0 @ x0  # System evolution
    x = x.reshape([param.nbVarX, param.nbData], order='F')
    f, J = f_reach(x[:,tl], param)  # Forward kinematics and Jacobian for end-effectors
    fc, Jc = f_reach_CoM(x, param)  # Forward kinematics and Jacobian for center of mass
#    print(Qc.shape)
#    print(fc.flatten('F'))

    du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + Su0.T @ Jc.T @ Qc @ Jc @ Su0 + R) @ \
         (-Su.T @ J.T @ Q @ f.flatten('F') - \
         Su0.T @ Jc.T @ Qc @ fc.flatten('F') - \
         u * param.r)

    # Estimate step size with line search method
    alpha = 1
    cost0 = f.flatten('F').T @ Q @ f.flatten('F') + np.linalg.norm(u)**2 * param.r  # Cost
    while True:
        utmp = u + du * alpha
        xtmp = Su0 @ utmp + Sx0 @ x0  # System evolution
        xtmp = xtmp.reshape([param.nbVarX, param.nbData], order='F')
        ftmp, _ = f_reach(xtmp[:,tl], param)
        fctmp, _ = f_reach_CoM(xtmp, param)

        # for end-effectors and CoM
        cost = ftmp.flatten('F').T @ Q @ ftmp.flatten('F') + np.linalg.norm(utmp)**2 * param.r  # Cost
        if cost < cost0 or alpha < 1e-3:
            print("Iteration {}, cost: {}, alpha: {}".format(i, cost, alpha))
            break # Stop iLQR when solution is reached

        alpha *= .5

    u = u + du * alpha
    if np.linalg.norm(du * alpha) < 1e-2:
        break

#  Plot state space
tl = np.array([0, tl.item()])

plt.figure(figsize=(15, 9))
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

# Plot bimanual robot
ftmp = fkin0(x[:,0], param)
plt.plot(ftmp[0], ftmp[1], c='black', linewidth=4, alpha=.2)

ftmp = fkin0(x[:,-1], param)
plt.plot(ftmp[0], ftmp[1], c='black', linewidth=4, alpha=.6)

# Plot CoM
fc = fkin_CoM(x, param)  # Forward kinematics for center of mass
plt.plot(fc[0,0], fc[1,0], c='black', marker='o', linewidth=0,
         markersize=np.sqrt(90), markerfacecolor='none', markeredgewidth=3, alpha=.4)  # Plot CoM
plt.plot(fc[0,-1], fc[1,-1], c='black', marker='o', linewidth=0,
         markersize=np.sqrt(90), markerfacecolor='none', markeredgewidth=3, alpha=.6)  # Plot CoM

# Plot end-effectors targets
for t in range(param.nbPoints):
    plt.plot(param.Mu[0,t], param.Mu[1,t], marker='o', c='red', markersize=14)

# Plot CoM target
plt.plot(param.MuCoM[0], param.MuCoM[1], c='red', marker='o', linewidth=0,
         markersize=np.sqrt(90), markerfacecolor='none', markeredgewidth=2, alpha=.8)

# Plot end-effectors paths
ftmp = fkin(x, param)
plt.plot(ftmp[0,:], ftmp[1,:], c="black", marker="o", markevery=[0] + tl.tolist())
plt.plot(ftmp[2,:], ftmp[3,:], c="black", marker="o", markevery=[0] + tl.tolist())
plt.show()


