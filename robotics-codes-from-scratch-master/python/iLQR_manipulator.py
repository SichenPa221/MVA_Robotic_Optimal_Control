'''
iLQR applied to a planar manipulator for a viapoints task (batch formulation)

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Jérémy Maceiras <jeremy.maceiras@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np

# Helper functions
# ===============================

# Logarithmic map for R^2 x S^1 manifold
def logmap(f, f0):
	position_error = f[:2,:] - f0[:2,:]
	orientation_error = np.imag(np.log(np.exp(f0[-1,:]*1j).conj().T * np.exp(f[-1,:]*1j).T)).conj()
	diff = np.vstack([position_error, orientation_error])
	return diff

# Forward kinematics for end-effector (in robot coordinate system)
def fkin(x, param):
	L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
	f = np.vstack([
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

# Residual and Jacobian for a viapoints reaching task (in object coordinate system)
def f_reach(x, param):
	f = logmap(fkin(x, param), param.Mu)

	J = np.zeros([param.nbPoints * param.nbVarF, param.nbPoints * param.nbVarX])
	for t in range(param.nbPoints):
		f[:2,t] = param.A[:,:,t].T @ f[:2,t] # Object oriented residual
		Jtmp = Jkin(x[:,t], param)
		Jtmp[:2] = param.A[:,:,t].T @ Jtmp[:2] # Object centered Jacobian
		
		if param.useBoundingBox:
			for i in range(2):
				if abs(f[i,t]) < param.sz[i]:
					f[i,t] = 0
					Jtmp[i] = 0
				else:
					f[i,t] -= np.sign(f[i,t]) * param.sz[i]
		
		J[t*param.nbVarF:(t+1)*param.nbVarF, t*param.nbVarX:(t+1)*param.nbVarX] = Jtmp
	return f, J

## Parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-2 # Time step length
param.nbData = 50 # Number of datapoints
param.nbIter = 100 # Maximum number of iterations for iLQR
param.nbPoints = 2 # Number of viapoints
param.nbVarX = 3 # State space dimension (x1,x2,x3)
param.nbVarU = 3 # Control space dimension (dx1,dx2,dx3)
param.nbVarF = 3 # Objective function dimension (f1,f2,f3, with f3 as orientation)
param.l = [2, 2, 1] # Robot links lengths
param.sz = [.2, .3] # Size of objects
param.r = 1e-6 # Control weight term
param.Mu = np.array([[2, 1, -np.pi/6], [3, 2, -np.pi/3]]).T # Viapoints 
param.A = np.zeros([2, 2, param.nbPoints]) # Object orientation matrices
param.useBoundingBox = True # Consider bounding boxes for reaching cost
for i in range(2):
	print(i)


# Main program
# ===============================

# Object rotation matrices
for t in range(param.nbPoints):
	orn_t = param.Mu[-1,t]
	param.A[:,:,t] = np.asarray([
		[np.cos(orn_t), -np.sin(orn_t)],
		[np.sin(orn_t), np.cos(orn_t)]
	])

# Precision matrix
Q = np.eye(param.nbVarF * param.nbPoints)
print('Q',Q)

# Control weight matrix
R = np.eye((param.nbData-1) * param.nbVarU) * param.r

# Time occurrence of viapoints
tl = np.linspace(0, param.nbData, param.nbPoints+1)
tl = np.rint(tl[1:]).astype(np.int64) - 1
idx = np.array([i + np.arange(0,param.nbVarX,1) for i in (tl*param.nbVarX)]).flatten() 

# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([
	np.zeros([param.nbVarX, param.nbVarX*(param.nbData-1)]), 
	np.tril(np.kron(np.ones([param.nbData-1, param.nbData-1]), np.eye(param.nbVarX) * param.dt))
]) 
Sx0 = np.kron(np.ones(param.nbData), np.eye(param.nbVarX)).T
Su = Su0[idx,:] # We remove the lines that are out of interest

print(Sx0.shape)
# iLQR
# ===============================

u = np.zeros(param.nbVarU * (param.nbData-1)) # Initial control command
x0 = np.array([3*np.pi/4, -np.pi/2, -np.pi/4]) # Initial state
for i in range(param.nbIter):
	x = Su0 @ u + Sx0 @ x0 # System evolution
	print(x.shape)
	x = x.reshape([param.nbVarX, param.nbData], order='F')
	print(x.shape)
	f, J = f_reach(x[:,tl], param) # Residuals and Jacobians
	du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + R) @ (-Su.T @ J.T @ Q @ f.flatten('F') - u * param.r) # Gauss-Newton update
	# Estimate step size with backtracking line search method
	alpha = 0.1
	cost0 = f.flatten('F').T @ Q @ f.flatten('F') + np.linalg.norm(u)**2 * param.r # Cost
	while True:
		utmp = u + du * alpha
		xtmp = Su0 @ utmp + Sx0 @ x0 # System evolution
		xtmp = xtmp.reshape([param.nbVarX, param.nbData], order='F')
		ftmp,_ = f_reach(xtmp[:,tl], param) # Residuals 
		cost = ftmp.flatten('F').T @ Q @ ftmp.flatten('F') + np.linalg.norm(utmp)**2 * param.r # Cost
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

# Get points of interest
f00 = fkin0(x[:,0], param)
f01 = fkin0(x[:,tl[0]], param)
f02 = fkin0(x[:,tl[1]], param)
f = fkin(x, param)

plt.plot(f00[0,:], f00[1,:], c='black', linewidth=5, alpha=.2)
plt.plot(f01[0,:], f01[1,:], c='black', linewidth=5, alpha=.4)
plt.plot(f02[0,:], f02[1,:], c='black', linewidth=5, alpha=.6)
plt.plot(f[0,:], f[1,:], c='black', marker='o', markevery=[0]+tl.tolist()) 

# Plot bounding box or viapoints
ax = plt.gca()
color_map = ['deepskyblue', 'darkorange']

for t in range(param.nbPoints):
    rect_origin = param.Mu[:2, t] - param.A[:, :, t] @ np.array(param.sz)
    rect_orn = param.Mu[-1, t]
    
    # Create a rectangle with no rotation
    rect = patches.Rectangle(rect_origin, param.sz[0]*2, param.sz[1]*2, color=color_map[t])
    
    # Apply the rotation using Affine2D from matplotlib.transforms
    t_rot = transforms.Affine2D().rotate_around(rect_origin[0], rect_origin[1], rect_orn)
    rect.set_transform(t_rot + ax.transData)

    ax.add_patch(rect)
	#plt.scatter(param.Mu[0,t], param.Mu[1,t], s=100, marker='X', c=color_map[t])

plt.show()
