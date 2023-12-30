'''
Batch iLQR applied on a planar manipulator for a tracking problem 
involving the center of mass (CoM) and the end-effector

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Teng Xue <teng.xue@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Helper functions
# ===============================


# Forward kinematics for end-effector (in robot coordinate system)
def fkin(x, param):
	L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
	f = np.vstack([
		param.l @ np.cos(L @ x),
		param.l @ np.sin(L @ x)
	]) 
	return f

# Forward kinematics for all joints (in robot coordinate system)
def fkin0(x, param): 
	L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
	f = np.vstack([
		L @ np.diag(param.l) @ np.cos(L @ x),
		L @ np.diag(param.l) @ np.sin(L @ x)
	])
	f = np.hstack([np.zeros([2,1]), f])
	return f

# Jacobian with analytical computation (for single time step)
def Jkin(x, param):
	L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
	J = np.vstack([
		-np.sin(L @ x).T @ np.diag(param.l) @ L,
		 np.cos(L @ x).T @ np.diag(param.l) @ L
	])
	return J

# Residual and Jacobian for a viapoints reaching task (in object coordinate system)
def f_reach(x, param):
	f = fkin(x, param) - param.Mu.reshape((-1,1))
	J = np.zeros([param.nbPoints * param.nbVarF, param.nbPoints * param.nbVarX])
	for t in range(param.nbPoints):
		J[t*param.nbVarF:(t+1)*param.nbVarF, t*param.nbVarX:(t+1)*param.nbVarX] = Jkin(x[:,t], param)
	return f, J

# Forward kinematics for center of mass (in robot coordinate system, with mass located at the joints)
def fkin_CoM(x, param):
    L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
    f = np.vstack((param.l @ L @ np.cos(L @ x),
                   param.l @ L @ np.sin(L @ x))) / param.nbVarX
    return f

# Jacobian for center of mass (in robot coordinate system, with mass located at the joints)
def Jkin_CoM(x, param):
    L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
    J = np.vstack((-np.sin(L @ x).T @ L @ np.diag(param.l @ L) ,
                    np.cos(L @ x).T @ L @ np.diag(param.l @ L))) / param.nbVarX
    return J

# Residual and Jacobian for center of mass
def f_reach_CoM(x, param):
    f = fkin_CoM(x, param) - np.tile(param.MuCoM.reshape((-1,1)), [1,np.size(x,1)])
    J = np.zeros((np.size(x,1) * param.nbVarF, np.size(x,1) * param.nbVarX))
    for t in range(np.size(x,1)):
        Jtmp = Jkin_CoM(x[:,t], param)
        if param.useBoundingBox:
            for i in range(1): # Only x direction matters
                if abs(f[i,t]) < param.szCoM:
                    f[i,t] = 0
                    Jtmp[i] = 0
                else:
                    f[i,t] -= np.sign(f[i,t]) * param.szCoM
        J[t*param.nbVarF:(t+1)*param.nbVarF, t*param.nbVarX:(t+1)*param.nbVarX] = Jtmp
    return f, J


# Parameters
param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-1 # Time step length
param.nbData = 10 # Number of datapoints
param.nbIter = 50 # Maximum number of iterations for iLQR
param.nbPoints = 1 # Number of viapoints
param.nbVarX = 5 # State space dimension (x1,x2,x3)
param.nbVarU = 5 # Control space dimension (dx1,dx2,dx3)
param.nbVarF = 2 # Objective function dimension (f1,f2,f3, with f3 as orientation)
param.l = [2, 2, 2, 2, 2] # Robot links lengths
param.szCoM = .6 # Size of the center of mass
param.useBoundingBox = True
param.r = 1e-5 # Control weight term
param.Mu = np.array([3.5, 4]) # Target
param.MuCoM = np.array([.4, 0]) # desired position of the center of mass


# Main program
# ===============================

# Task parameters
# ===============================
# Regularization matrix
R = np.identity((param.nbData - 1) * param.nbVarU) * param.r

# Precision matrix
Q = np.identity(param.nbVarF * param.nbPoints)

# Precision matrix for CoM (by considering only horizonal CoM location)
Qc = np.kron(np.identity(param.nbData), np.diag([1E0, 0]))

# System parameters
# ===============================
# Time occurrence of viapoints
tl = np.linspace(0, param.nbData, param.nbPoints + 1)
tl = np.rint(tl[1:]).astype(np.int64) - 1
idx = np.array([i + np.arange(0,param.nbVarX,1) for i in (tl* param.nbVarX)])

u = np.zeros(param.nbVarU * (param.nbData - 1))  # Initial control command
a = .7
x0 = np.array([np.pi/2-a, 2*a, -a, 3*np.pi/4, 3*np.pi/4])  # Initial state (in joint space)

# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([np.zeros((param.nbVarX, param.nbVarX * (param.nbData - 1))),
                 np.tril(np.kron(np.ones((param.nbData - 1, param.nbData - 1)),
                 np.eye(param.nbVarX) * param.dt))])
Sx0 = np.kron(np.ones(param.nbData), np.identity(param.nbVarX)).T
Su = Su0[idx.flatten()]  # We remove the lines that are out of interest


# Solving iLQR
# ===============================
for i in range(param.nbIter):
    x = Su0 @ u + Sx0 @ x0
    x = x.reshape([param.nbVarX, param.nbData], order='F')
    f, J = f_reach(x[:,tl], param)
    fc, Jc = f_reach_CoM(x, param)
    du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + Su0.T @ Jc.T @ Qc @ Jc @ Su0 + R) @ \
                      (-Su.T @ J.T @ Q @ f.flatten('F') - Su0.T @ Jc.T @ Qc @ fc.flatten('F') - u * param.r)

    # Perform line search
    alpha = 1
    cost0 = f.flatten('F').T @ Q @ f.flatten('F') + fc.flatten('F').T @ Qc @ fc.flatten('F') + np.linalg.norm(u) * param.r

    while True:
        utmp = u + du * alpha
        xtmp = Su0 @ utmp + Sx0 @ x0
        xtmp = xtmp.reshape([param.nbVarX, param.nbData], order='F')
        ftmp, _ = f_reach(xtmp[:,tl], param)
        fctmp, _ = f_reach_CoM(xtmp, param)
        cost = ftmp.flatten('F').T @ Q @ ftmp.flatten('F') + fctmp.flatten('F').T @ Qc @ fctmp.flatten('F') + np.linalg.norm(utmp)**2 * param.r

        if cost < cost0 or alpha < 1e-3:
            u = utmp
            print("Iteration {}, cost: {}, alpha: {}".format(i, cost, alpha))
            break

        alpha /= 2

# Plotting
# ===============================
plt.figure()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

# plot ground
plt.plot([-1, 3], [0, 0], linestyle='-', c=[.2, .2, .2], linewidth=2)

# Get points of interest
f00 = fkin0(x[:,0], param)
fT0 = fkin0(x[:,-1], param)
fc = fkin_CoM(x, param)

plt.plot(f00[0,:], f00[1,:], c=[.8, .8, .8], linewidth=4, linestyle='-')
plt.plot(fT0[0,:], fT0[1,:], c=[.4, .4, .4], linewidth=4, linestyle='-')

#plot CoM
plt.plot(fc[0, 0], fc[1, 0], c=[.5, .5, .5], marker="o", markeredgewidth=4, markersize=8, markerfacecolor='white')
plt.plot(fc[0, -1], fc[1, -1], c=[.2, .2, .2], marker="o", markeredgewidth=4, markersize=8, markerfacecolor='white')

#plot end-effector target
plt.plot(param.Mu[0], param.Mu[1], marker="o", markersize=8, c="r")

# Plot bounding box or via-points
ax = plt.gca()
for i in range(param.nbPoints):
    if param.useBoundingBox:
        rect_origin = param.MuCoM + np.array([0, 3.5]) - np.array([param.szCoM, 3.5])
        rect = patches.Rectangle(rect_origin, param.szCoM * 2, 3.5 * 2,
                                 facecolor=[.8, 0, 0], alpha=0.1, edgecolor=None)
        ax.add_patch(rect)
plt.show()
