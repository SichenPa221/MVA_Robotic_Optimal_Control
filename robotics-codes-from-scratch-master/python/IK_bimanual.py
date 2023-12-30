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


param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-2 # Time step length
param.nbData = 30 # Number of datapoints
param.nbVarX = 5 # State space dimension ([q1,q2,q3] for left arm, [q1,q4,q5] for right arm)
param.nbVarF = 4 # Task space dimension ([x1,x2] for left end-effector, [x3,x4] for right end-effector)
param.l = np.ones(param.nbVarX) * 2 # Robot links lengths
param.Mu = np.array([-1, -1.5, 4, 2]) # Target point for end-effectors
x = np.array([np.pi/2, np.pi/2, np.pi/3, -np.pi/2, -np.pi/3]) # Initial pose

## Inverse kinematics (IK)
# ===============================

plt.scatter(param.Mu[0], param.Mu[1], color='r', marker='.', s=10**2) #Plot left target
plt.scatter(param.Mu[2], param.Mu[3], color='r', marker='.', s=10**2) #Plot left target
for t in range(param.nbData):
	f = fkin(x, param) #Forward kinematics (for end-effector)
	J = Jkin(x, param) #Jacobian (for end-effector)
	x += np.linalg.pinv(J) @ (param.Mu - f[:,0]) * 20 * param.dt #Update state 
	f_rob = fkin0(x, param) #Forward kinematics (for all articulations, including end-effector)
	plt.plot(f_rob[0,:], f_rob[1,:], color=str(1-t/param.nbData), linewidth=1) #Plot robot

plt.axis('off')
plt.axis('equal')
plt.show()


