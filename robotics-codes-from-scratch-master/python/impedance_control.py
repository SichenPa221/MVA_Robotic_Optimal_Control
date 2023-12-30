'''
Impedance controller in joint/task space

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Sylvain Calinon <https://calinon.ch> and
Amirreza Razmjoo <amirreza.razmjoo@idiap.ch>

This file is part of RCFS <https://rcfs.ch/>
License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Helper functions
# ===============================

# NB: Because of the use of matplotlib.animation, we need to set param as a global variable
# Initialization of the plot
def init():
    global param
    ax.set_xlim(-np.sum(param.l) - 0.1, np.sum(param.l) + 0.1)
    ax.set_ylim(-np.sum(param.l) - 0.1, np.sum(param.l) + 0.1)
    return ln1, ln2

# NB: Because of the use of matplotlib.animation, we need to set param as a global variable
# Updating the values in the plot
def animate(i):
    global param
    f = fkin0(x_anim[0:param.nbVarX,i], param)
    ln1.set_data(f[0,:], f[1,:])
    ln2.set_data(f[0,:], f[1,:])
    return ln1, ln2

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
		 np.cos(L @ x).T @ np.diag(param.l) @ L,
		 np.ones([1,param.nbVarX])
	])
	return J

def computeGCML(x, param):
	# Auxiliary matrices
	l = np.reshape(param.l, [1, param.nbVarX])
	L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
	m = np.reshape(param.m, [1, param.nbVarX])
	Lm = np.triu(np.ones([m.shape[1], m.shape[1]])) * np.repeat(m, m.shape[1],0)

#	# Elementwise computation of G, C, and M
#	G = np.zeros(param.nbVarX)
#	M = np.zeros([param.nbVarX, param.nbVarX])
#	C =  np.zeros([param.nbVarX, param.nbVarX])
#	for k in range(param.nbVarX):
#		G[k] = -sum(m[0,k:]) * param.gravity * l[0,k] * np.cos(L[k,:] @ x[:param.nbVarX])
#		for i in range(param.nbVarX):
#			S = sum(m[0,k:param.nbVarX] * np.heaviside(np.array(range(k, param.nbVarX)) - i, 1))
#			M[k,i] = l[0,k] * l[0,i] * np.cos(L[k,:] @ x[:param.nbVarX] - L[i,:] @ x[:param.nbVarX]) * S
#			C[k,i] = -l[0,k] * l[0,i] * np.sin(L[k,:] @ x[:param.nbVarX] - L[i,:] @ x[:param.nbVarX]) * S
	
	# Computation in matrix form of G, C, and M
	G = -np.sum(Lm,1) * param.l * np.cos(L @ x[:param.nbVarX]) * param.gravity
	C = -(l.T * l) * np.sin(np.reshape(L @ x[:param.nbVarX], [param.nbVarX,1]) - L @ x[:param.nbVarX]) * (Lm**.5 @ ((Lm**.5).T))
	M = (l.T * l) * np.cos(np.reshape(L @ x[:param.nbVarX], [param.nbVarX,1]) - L @ x[:param.nbVarX]) * (Lm**.5 @ ((Lm**.5).T))
	
	G = L.T @ G
	C = L.T @ C
	M = L.T @ M @ L
	
	return G,C,M,L

def inverse_dynamics(x, ddx, param):
	G,C,M,L = computeGCML(x, param)
#	u = M @ ddx - G - C @ (L @ x[param.nbVarX:])**2 + x[param.nbVarX:] * param.damping # With gravity, Coriolis and viscous friction compensation models
	u = M @ ddx - G - C @ (L @ x[param.nbVarX:])**2 # With gravity and Coriolis models 
	return u

def fdyn(x, u, param):
	G,C,M,L = computeGCML(x, param)
	ddx = np.linalg.inv(M) @ (u + G + C @ (L @ x[param.nbVarX:])**2 - x[param.nbVarX:] * param.damping)
	return ddx


# General parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.dt = 1E-2 # Time step length
param.nbData = 500 # Number of datapoints
param.nbVarX = 3 # State space dimension (x1,x2,x3)
param.nbVarU = 3 # Control space dimension (dx1,dx2,dx3)
param.nbVarF = 3 # Objective function dimension (f1,f2,f3, with f3 as orientation)
param.l = np.array([1, 1, 1]) # Robot links lengths
param.m = np.array([1, 1, 1]) # Robot links masses
param.damping = 20.0 # Viscous friction
param.gravity = 9.81 # Gravity


# Initialization
# ===============================
x = np.zeros(2*param.nbVarX) # State (position and velocity)
x_anim = np.zeros([2*param.nbVarX, param.nbData]) # Array of states to replay animation 

x_target = np.array([-np.pi/3, -np.pi/2, np.pi/3]) # Target in joint space
f_target = fkin(np.array([-np.pi/4, -np.pi/2, np.pi/4]), param) # Target in task space

kP = 400.0 # Stiffness gain in joint space
kV = 10.0 # Damping gain in joint space

KP = np.diag([4E2, 4E2, 4E2]) # Stiffness gain in task space
KV = np.diag([1E1, 1E1, 1E1]) # Damping gain in task space

# Simulation
for t in range(param.nbData):
#	xtmp = np.append(x[:param.nbVarX], np.zeros(param.nbVarX))
#	ug = inverse_dynamics(xtmp, np.zeros(param.nbVarX), param) # Torques for gravity compensation

	ug = inverse_dynamics(x, np.zeros(param.nbVarX), param) # Torques for gravity and Coriolis force compensation
	
#	# Impedance controller in joint space
#	u = kP * (x_target - x[:param.nbVarX]) - kV * x[param.nbVarX:] + ug # Torque commands
	
	# Impedance controller in task space
	f = fkin(x[:param.nbVarX], param)
	J = Jkin(x[:param.nbVarX], param)
	df = J @ x[param.nbVarX:]
	u = J.T @ (KP @ logmap(f_target, f) - KV @ df) + ug # Torque commands
	
	ddx = fdyn(x, u, param) # Compute accelerations
	x += np.append(x[param.nbVarX:] + 0.5 * ddx * param.dt, ddx) * param.dt # Update state
	x_anim[:,t] = x # Log data


# Plot
# ===============================
fig, ax = plt.subplots()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(f_target[0], f_target[1], color='r', marker='.', s=15**2) #Plot target
ln1, = plt.plot([], [], '-')
ln2, = plt.plot([], [], 'o-', linewidth=2, markersize=5, c="black")
ani = animation.FuncAnimation(fig, animate, x_anim.shape[1], init_func=init, interval = param.dt * 500, blit= True, repeat = False)
plt.show()

