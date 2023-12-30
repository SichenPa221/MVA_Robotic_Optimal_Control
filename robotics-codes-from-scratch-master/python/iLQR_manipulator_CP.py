'''
Batch iLQR with control primitives applied on a manipulator example

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Jérémy Maceiras <jeremy.maceiras@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import special
from math import factorial

# Helper functions
# ===============================

# Logarithmic map for R^2 x S^1 manifold
def logmap(f,f0):
	position_error = f[:,:2] - f0[:,:2]
	orientation_error = np.imag(np.log( np.exp( f0[:,-1]*1j ).conj().T * np.exp(f[:,-1]*1j).T )).conj().reshape((-1,1))
	error = np.hstack(( position_error , orientation_error ))
	return error

# Forward kinematics for E-E
def fkin(param,x):
	x = x.T
	A = np.tril(np.ones([param.nbVarX,param.nbVarX]))
	f = np.vstack((param.linkLengths @ np.cos(A @ x), 
				   param.linkLengths @ np.sin(A @ x), 
				   np.mod(np.sum(x,0)+np.pi, 2*np.pi) - np.pi)) #x1,x2,o (orientation as single Euler angle for planar robot)
	return f.T

# Forward Kinematics for all joints
def fkin0(param,x):
	T = np.tril(np.ones([param.nbVarX,param.nbVarX]))
	T2 = np.tril(np.matlib.repmat(param.linkLengths,len(x),1))
	f = np.vstack(( 
		T2 @ np.cos(T@x),
		T2 @ np.sin(T@x)
	)).T
	f = np.vstack(( 
		np.zeros(2),
		f
	))
	return f

# Jacobian with analytical computation (for single time step)
def Jkin(param,x):
	T = np.tril( np.ones((len(x),len(x))) )
	J = np.vstack((
		-np.sin(T@x).T @ np.diag(param.linkLengths) @ T,
		np.cos(T@x).T @ np.diag(param.linkLengths) @ T,
		np.ones(len(x))
	))
	return J

# Residual and Jacobian
def f_reach(param,x):
	f = logmap(fkin(param,x),param.mu)
	J = np.zeros(( len(x) * param.nbVarF , len(x) * param.nbVarX ))

	for t in range(x.shape[0]):
		f[t,:2] = param.A[t].T @ f[t,:2] # Object oriented fk
		Jtmp = Jkin(param,x[t])
		Jtmp[:2] = param.A[t].T @ Jtmp[:2] # Object centered jacobian

		if param.useBoundingBox:
			for i in range(2):
				if abs(f[t,i]) < param.sizeObj[i]:
					f[t,i] = 0
					Jtmp[i]=0
				else:
					f[t,i] -=  np.sign(f[t,i]) * param.sizeObj[i]
		J[t*param.nbVarF:(t+1)*param.nbVarF, t*param.nbVarX:(t+1)*param.nbVarX] = Jtmp
	return f,J

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

# General param parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-2 # Time step length
param.nbData = 50 # Number of datapoints
param.nbIter = 100 # Maximum number of iterations for iLQR
param.nbPoints = 2 # Number of viapoints
param.nbVarX = 3 # State space dimension (x1,x2,x3)
param.nbVarU = 3 # Control space dimension (dx1,dx2,dx3)
param.nbVarF = 3 # Objective function dimension (f1,f2,f3, with f3 as orientation)
param.linkLengths = [2,2,1] # Robot links lengths
param.sizeObj = [.2,.3] # Size of objects
param.r = 1e-6 # Control weight term
param.mu = np.asarray([[2,1,-np.pi/6], [3,2,-np.pi/3]]).T # Viapoints 
param.useBoundingBox = True # Consider bounding boxes for reaching cost
param.nbFct = 5 # Number of basis functions
param.basisName = "PIECEWISE"

# Task parameters
# ===============================

# Targets
param.mu = np.asarray([ # x, y, orientation
	[2,1,-np.pi/2],	
	[3,1,-np.pi/2]
])

# Transformation matrices
param.A = np.zeros( (param.nbPoints,2,2) )
for i in range(param.nbPoints):
	orn_t = param.mu[i,-1]
	param.A[i,:,:] = np.asarray([
		[np.cos(orn_t) , -np.sin(orn_t)],
		[np.sin(orn_t) , np.cos(orn_t)]
	])

# Regularization matrix
R = np.identity( (param.nbData-1) * param.nbVarU ) * param.r

# Precision matrix
Q = np.identity( param.nbVarF  * param.nbPoints)

# System parameters
# ===============================

# Time occurrence of viapoints
tl = np.linspace(0,param.nbData,param.nbPoints+1)
tl = np.rint(tl[1:]).astype(np.int64)-1
idx = np.array([ i + np.arange(0,param.nbVarX,1) for i in (tl* param.nbVarX)]) 

u = np.zeros(param.nbVarU * (param.nbData-1)) # Initial control command
x0 = np.array([3*np.pi/2, -np.pi/2, - np.pi/4]) # Initial state (in joint space)

# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([np.zeros((param.nbVarX, param.nbVarX*(param.nbData-1))), 
      np.tril(np.kron(np.ones((param.nbData-1, param.nbData-1)), np.eye(param.nbVarX)*param.dt))]) 
Sx0 = np.kron( np.ones(param.nbData) , np.identity(param.nbVarX) ).T
Su = Su0[idx.flatten()] # We remove the lines that are out of interest

# Basis functions
# ================

functions = {
    "PIECEWISE": build_phi_piecewise,
    "RBF": build_phi_rbf,
    "BERNSTEIN": build_phi_bernstein,
    "FOURIER": build_phi_fourier
}
phi = functions[param.basisName](param.nbData-1,param.nbFct)
PSI = np.kron(phi,np.identity(param.nbVarU))

# Solving iLQR
# ===============================

for i in range( param.nbIter ):
	x = np.real(Su0 @ u + Sx0 @ x0)
	x = x.reshape( (  param.nbData , param.nbVarX) )

	f, J = f_reach(param,x[tl])
	dw = np.linalg.inv(PSI.T @ Su.T @ J.T @ Q @ J @ Su @ PSI + PSI.T @ R @ PSI) @ (-PSI.T @ Su.T @ J.T @ Q @ f.flatten() - PSI.T @ u * param.r)
	du = PSI @ dw
	# Perform line search
	alpha = 1
	cost0 = f.flatten() @ Q @ f.flatten() + np.linalg.norm(u) * param.r
	
	while True:
		utmp = u + du * alpha
		xtmp = np.real(Su0 @ utmp + Sx0 @ x0)
		xtmp = xtmp.reshape( (  param.nbData , param.nbVarX) )
		ftmp, _ = f_reach(param,xtmp[tl])
		cost = ftmp.flatten() @ Q @ ftmp.flatten() + np.linalg.norm(utmp) * param.r
		
		if cost < cost0 or alpha < 1E-3:
			u = utmp
			print("Iteration {}, cost: {}, alpha: {}".format(i,cost,alpha))
			break

		alpha /=2

	if np.linalg.norm( alpha * du ) < 1E-2:
		break

# Plotting
# ===============================

plt.figure()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

# Get points of interest
f = fkin(param,x)
f00 = fkin0(param,x[0])
f10 = fkin0(param,x[tl[0]])
fT0 = fkin0(param,x[-1])
u = u.reshape((-1,param.nbVarU))

plt.plot(f00[:,0], f00[:,1], c='black',linewidth=5,alpha=.2)
plt.plot(f10[:,0], f10[:,1], c='black',linewidth=5,alpha=.4)
plt.plot(fT0[:,0], fT0[:,1], c='black',linewidth=5,alpha=.6)

plt.plot(f[:,0],f[:,1],c="black",marker="o",markevery=[0]+tl.tolist()) #,label="Trajectory"2

# Plot bounding box or via-points
ax = plt.gca()
color_map = ["deepskyblue","darkorange"]
for i in range(param.nbPoints):
	
	if param.useBoundingBox:
		rect_origin = param.mu[i,:2] - param.A[i]@np.array(param.sizeObj)
		rect_orn = param.mu[i,-1]

		rect = patches.Rectangle(rect_origin,param.sizeObj[0]*2,param.sizeObj[1]*2,np.degrees(rect_orn),color=color_map[i])
		ax.add_patch(rect)
	else:
		plt.scatter(param.mu[i,0],param.mu[i,1],s=100,marker="X",c=color_map[i])

fig,axs = plt.subplots(7,1)

axs[0].plot(f[:,0],c='black')
axs[0].set_ylabel("$f(x)_1$")
axs[0].set_xticks([0,param.nbData])
axs[0].set_xticklabels(["0","T"])
for i in range(param.nbPoints):
    axs[0].scatter(tl[i],param.mu[i,0],c='blue')

axs[1].plot(f[:,1],c='black')
axs[1].set_ylabel("$f(x)_2$")
axs[1].set_xticks([0,param.nbData])
axs[1].set_xticklabels(["0","T"])
for i in range(param.nbPoints):
    axs[1].scatter(tl[i],param.mu[i,1],c='blue')

axs[2].plot(f[:,2],c='black')
axs[2].set_ylabel("$f(x)_3$")
axs[2].set_xticks([0,param.nbData])
axs[2].set_xticklabels(["0","T"])
for i in range(param.nbPoints):
    axs[2].scatter(tl[i],param.mu[i,2],c='blue')

axs[3].plot(u[:,0],c='black')
axs[3].set_ylabel("$u_1$")
axs[3].set_xticks([0,param.nbData-1])
axs[3].set_xticklabels(["0","T-1"])

axs[4].plot(u[:,1],c='black')
axs[4].set_ylabel("$u_2$")
axs[4].set_xticks([0,param.nbData-1])
axs[4].set_xticklabels(["0","T-1"])

axs[5].plot(u[:,2],c='black')
axs[5].set_ylabel("$u_3$")
axs[5].set_xticks([0,param.nbData-1])
axs[5].set_xticklabels(["0","T-1"])

axs[6].set_ylabel("$\phi_k$")
axs[6].set_xticks([0,param.nbData-1])
axs[6].set_xticklabels(["0","T-1"])
for i in range(param.nbFct):
    axs[6].plot(phi[:,i])
axs[6].set_xlabel("$t$")

plt.show()
