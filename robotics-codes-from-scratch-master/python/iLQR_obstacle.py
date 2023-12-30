'''
Batch iLQR with obstacle avoidance

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Jérémy Maceiras <jeremy.maceiras@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Residual and Jacobian for a via point reaching task
def f_reach(x,param):
    f = x - param.Mu[:,:2]
    J = np.identity(x.shape[1]*x.shape[0])
    return f, J

# Residual and Jacobian for an obstacle avoidance task
def f_avoid(x,param):
    f=[]
    idx=[]
    idt=[]
    J=np.zeros((0,0))

    for i in range(x.shape[0]): # Go through all the via points
        for j in range(param.Obs.shape[0]): # Go through all the obstacles
            e = param.U_obs[j].T @ (x[i] - param.Obs[j][:2])
            ftmp = 1 - e.T @ e

            if ftmp > 0:
                f.append(ftmp)
                
                Jtmp = -(param.U_obs[j] @ e).T.reshape((-1,1))
                J2 = np.zeros(( J.shape[0] + Jtmp.shape[0] , J.shape[1] + Jtmp.shape[1] ))
                J2[:J.shape[0],:J.shape[1]] = J
                J2[-Jtmp.shape[0]:,-Jtmp.shape[1]:] = Jtmp
                J = J2 # Numpy does not provide a 'blockdiag' function...

                idx.append( i*param.nbVarU + np.array(range(param.nbVarU)) )
                idt.append(i)
    f = np.array(f)
    idx = np.array(idx)
    idt = np.array(idt)
    return f, J.T, idx, idt

# General parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-2 # Time step length
param.nbData = 100 # Number of datapoints
param.nbIter = 300 # Maximum number of iterations for iLQR
param.nbObstacles = 2 # Number of obstacles
param.nbPoints = 1 # Number of viapoints
param.nbVarX = 2 # State space dimension (x1,x2,x3)
param.nbVarU = 2 # Control space dimension (dx1,dx2,dx3)
param.sizeObstacle = [.4,.6] # Size of objects
param.r = 1e-3 # Control weight term
param.Q_track = 1e2 # Reaching task weight term
param.Q_avoid = 1e0 # Obstacle avoidance task weight term

# Task parameters
# ===============================

# Target
param.Mu = np.array([
    [ 3 , 3 , np.pi/6 ]
    ]) # Viapoint [x1,x2,o]

# Obstacles
param.Obs = np.array([
[1,0.6,np.pi/4],        # [x1,x2,o]
[2,2.5,-np.pi/6]        # [x1,x2,o]
])

param.A_obs = np.zeros(( param.nbObstacles , 2 , 2 ))
param.S_obs = np.zeros(( param.nbObstacles , 2 , 2 ))
param.Q_obs = np.zeros(( param.nbObstacles , 2 , 2 ))
param.U_obs = np.zeros(( param.nbObstacles , 2 , 2 )) # Q_obs[t] = U_obs[t].T @ U_obs[t]
for i in range(param.nbObstacles):
    orn_t = param.Obs[i][-1]
    param.A_obs[i] = np.array([              # Orientation in matrix form
        [ np.cos(orn_t) , -np.sin(orn_t)  ],
        [ np.sin(orn_t) , np.cos(orn_t)]
    ])

    param.S_obs[i] = param.A_obs[i] @ np.diag(param.sizeObstacle)**2 @ param.A_obs[i].T # Covariance matrix
    param.Q_obs[i] = np.linalg.inv( param.S_obs[i] ) # Precision matrix
    param.U_obs[i] = param.A_obs[i] @ np.diag(1/np.array(param.sizeObstacle)) # "Square root" of param.Q_obs[i]

# Regularization matrix
R = np.identity( (param.nbData-1) * param.nbVarU ) * param.r

# System parameters
# ===============================

# Time occurrence of viapoints
tl = np.linspace(0,param.nbData,param.nbPoints+1)
tl = np.rint(tl[1:]).astype(np.int64)-1
idx = np.array([ i + np.arange(0,param.nbVarX,1) for i in (tl* param.nbVarX)]) 

u = np.zeros( param.nbVarU * (param.nbData-1) ) # Initial control command
x0 = np.zeros( param.nbVarX ) # Initial state

# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([np.zeros((param.nbVarX, param.nbVarX*(param.nbData-1))), 
np.tril(np.kron(np.ones((param.nbData-1, param.nbData-1)), np.eye(param.nbVarX)*param.dt))]) 
Sx0 = np.kron( np.ones(param.nbData) , np.identity(param.nbVarX) ).T
Su = Su0[idx.flatten()] # We remove the lines that are out of interest

# Solving iLQR
# ===============================

for i in range( param.nbIter ):
    x = Su0 @ u + Sx0 @ x0
    x = x.reshape( (  param.nbData , param.nbVarX) )

    f, J = f_reach(x[tl],param)  # Tracking objective
    f2, J2, id2 , _ = f_avoid(x,param)# Avoidance objective
    
    if len(id2) > 0: # Numpy does not allow zero sized array as Indices
        Su2 = Su0[id2.flatten()]
        du = np.linalg.inv(Su.T @ J.T @ J @ Su * param.Q_track + Su2.T @ J2.T @ J2 @ Su2 * param.Q_avoid + R) @ \
                (-Su.T @ J.T @ f.flatten() * param.Q_track - Su2.T @ J2.T @ f2.flatten() * param.Q_avoid - u * param.r)
    else: # It means that we have a collision free path
        du = np.linalg.inv(Su.T @ J.T @ J @ Su * param.Q_track + R) @ \
                (-Su.T @ J.T @ f.flatten() * param.Q_track - u * param.r)

    # Perform line search
    alpha = 1
    cost0 = np.linalg.norm(f.flatten())**2 * param.Q_track + np.linalg.norm(f2.flatten())**2 * param.Q_avoid + np.linalg.norm(u) * param.r

    while True:
        utmp = u + du * alpha
        xtmp = Su0 @ utmp + Sx0 @ x0
        xtmp = xtmp.reshape( (  param.nbData , param.nbVarX) )
        ftmp, _ = f_reach(xtmp[tl],param)
        f2tmp,_,_,_ = f_avoid(xtmp,param)
        cost = np.linalg.norm(ftmp.flatten())**2 * param.Q_track + np.linalg.norm(f2tmp.flatten())**2 * param.Q_avoid + np.linalg.norm(utmp) * param.r

        if cost < cost0 or alpha < 1e-3:
            u = utmp
            print("Iteration {}, cost: {}, alpha: {}".format(i,cost,alpha))
            break

        alpha /=2
    
    if np.linalg.norm(alpha * du) < 1e-2: # Early stop condition
        break

# Plotting
# ===============================

plt.figure()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

# Get trajectory
x = Su0 @ u + Sx0 @ x0
x = x.reshape( (param.nbData, param.nbVarX) )

plt.scatter(x[0,0], x[0,1], c='black', s=40)

# Plot targets
for i in range(param.nbPoints):
    xt = param.Mu[i]
    plt.scatter(xt[0], xt[1], c='red', s=40)

# Plot obstacles
al = np.linspace(-np.pi, np.pi, 50)
ax = plt.gca()
for i in range(param.nbObstacles):
    D,V = np.linalg.eig(param.S_obs[i])
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    msh = (R @ np.array([np.cos(al),np.sin(al)])).T + param.Obs[i][:2]
    p = patches.Polygon(msh, closed=True)
    ax.add_patch(p)

plt.plot(x[:,0], x[:,1], c='black')
plt.scatter(x[::10,0], x[::10,1], c='black', s=10)

plt.show()
