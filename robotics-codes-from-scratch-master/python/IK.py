'''
Inverse kinematics example

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt


# Logarithmic map for R^2 x S^1 manifold
def logmap(f, f0):
	diff = np.zeros(3)
	diff[:2] = f[:2] - f0[:2] # Position residual
	diff[2] = np.imag(np.log(np.exp(f0[-1]*1j).conj().T * np.exp(f[-1]*1j).T)).conj() # Orientation residual
	return diff
	
# Forward kinematics for end-effector (in robot coordinate system)
def fkin(x, param):
	L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
	f = np.stack([
		param.l @ np.cos(L @ x),
		param.l @ np.sin(L @ x),
		np.mod(np.sum(x,0)+np.pi, 2*np.pi) - np.pi
	]) # f1,f2,f3, where f3 is the orientation (single Euler angle for planar robot)
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
		 np.cos(L @ x).T @ np.diag(param.l) @ L,
		 np.ones([1,param.nbVarX])
	])
	return J

## Parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-2 # Time step length
param.nbData = 50 # Number of datapoints
param.nbVarX = 3 # State space dimension (x1,x2,x3)
param.nbVarU = 3 # Control space dimension (dx1,dx2,dx3)
param.nbVarF = 3 # Objective function dimension (position and orientation of the end-effector)
param.l = [2, 2, 1] # Robot links lengths

fh = np.array([3, 1, -np.pi/2]) # Desired target for the end-effector (position and orientation)
x = -np.ones(param.nbVarX) * np.pi / param.nbVarX # Initial robot pose
x[0] = x[0] + np.pi 

## Inverse kinematics (IK)
# ===============================

plt.scatter(fh[0], fh[1], color='r', marker='.', s=10**2) #Plot target
for t in range(param.nbData):
	f = fkin(x, param) # Forward kinematics (for end-effector)
	J = Jkin(x, param) # Jacobian (for end-effector)
#	x += np.linalg.pinv(J) @ (fh - f) * 10 * param.dt # Update state 
	x += np.linalg.pinv(J) @ logmap(fh, f) * 10 * param.dt # Update state 
	
	f_rob = fkin0(x, param) # Forward kinematics (for all articulations, including end-effector)
	plt.plot(f_rob[0,:], f_rob[1,:], color=str(1-t/param.nbData), linewidth=2) # Plot robot

plt.axis('off')
plt.axis('equal')
plt.show()
