'''
iLQR applied to a 2D point-mass system reaching a target while avoiding 
obstacles represented as Gaussian process implicit surfaces (GPIS)

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Yan Zhang <yan.zhang@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist as cdist

# Residuals f and Jacobians J for a viapoints reaching task (in object coordinate system)
def f_reach(xt,param):
    f = xt - param.Mu[:2, :]
    J = np.identity(xt.shape[1]*xt.shape[0])
    return f, J

# Covariance function in GPIS
def covFct(x1, x2, p, flag_noiseObs=0):
    # Thin plate covariance function (for 3D implicit shape)
    K = (1 / 12) * (2 * np.power(cdist(x1.T, x2.T), 3) 
        - 3 * p[0] * np.power(cdist(x1.T, x2.T), 2) 
        + np.power(p[0], 3)) # Kernel
    
    dK = np.zeros((x1.shape[1], x2.shape[1], x1.shape[0]))
    # Derivatives along x1
    dK[:,:,0] = (1 / 12) * (6 * np.multiply(cdist(x1.T, x2.T), 
        np.subtract(x1[0,:][:,None], x2[0,:])) 
        - 6 * p[0] * np.subtract(x1[0,:][:,None], x2[0,:]))
    # Derivatives along x2
    dK[:,:,1] = (1 / 12) * (6 * np.multiply(cdist(x1.T, x2.T), 
        np.subtract(x1[1,:][:,None], x2[1,:])) 
        - 6 * p[0] * np.subtract(x1[1,:][:,None], x2[1,:]))
    
    # # RBF covariance function
    # p = [1 / (5E-2), 1E-4, 1E-2]
    # K = p[1] * np.exp(-p[0] * np.power(cdist(x1.T, x2.T), 2)) # Kernel
    # # Derivatives along x1
    # dK[:,:,0] = -p[0] * p[1] * np.exp(-p[0] * np.power(cdist(x1.T, x2.T), 2)) 
    #     * np.subtract(x1[0,:][:,None], x2[0,:])
    # # Derivatives along x2
    # dK[:,:,1] = -p[0] * p[1] * np.exp(-p[0] * np.power(cdist(x1.T, x2.T), 2)) 
    #     * np.subtract(x1[1,:][:,None]·, x2[1,:])
    
    if flag_noiseObs==1:
        K += p[1] * np.eye(x1.shape[1], x2.shape[1])
        
    return K, dK

def GPIS(x, param):
    K, dK = covFct(x, param.x, param.p)
    f = (K @ np.linalg.inv(param.K) @ param.y.T).T # GPR with Mu=0
    # f = param.MuS + (K / param.K * (param.y - param.Mu2)')';
    J = np.concatenate([(dK[:,:,0] @ np.linalg.inv(param.K) @ (param.y - param.Mu2).T).T,
                        (dK[:,:,1] @ np.linalg.inv(param.K) @ (param.y - param.Mu2).T).T], axis=0) # Gradients
    
    # Reshape gradients
    a = np.maximum(f, 0) # Amplitude
    J = 1E2 * np.tile(np.tanh(a), [2,1]) * J / np.tile(np.sum(J**2, axis=0, keepdims=True)**.5, [2,1]) # Vector moving away from interior of shape
    return f, J

def f_avoid(x, param):
    ftmp, Jtmp = GPIS(x, param)
    f, id, idt = [], [], []
    J = np.zeros((0,0))
    for t in range(x.shape[1]):
        # Bounding boxes
        if ftmp[0][t] > 0:
            f.append(ftmp[0][t])
            Jtmp2 = Jtmp[:, t:(t+1)].T
            J2 = np.zeros(( J.shape[0] + Jtmp2.shape[0] , J.shape[1] + Jtmp2.shape[1] ))
            J2[:J.shape[0],:J.shape[1]] = J
            J2[-Jtmp2.shape[0]:,-Jtmp2.shape[1]:] = Jtmp2
            J = J2

            id.append(t*param.nbVarU + np.array(range(param.nbVarU)) )
            idt.append(t)

    f = np.array(f).reshape((-1, 1))
    id = np.array(id)
    idt = np.array(idt)     
    return f, J, id, idt


# General parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-2 # Time step length
param.nbData = 100 # Number of datapoints
param.nbIter = 300 # Maximum number of iterations for iLQR
param.nbPoints = 1 # Number of viapoints
param.nbVarX = 2 # State space dimension (x1,x2)
param.nbVarU = 2 # Control space dimension (dx1,dx2)
param.sz = [.2, .2] # Size of objects
param.sz2 = [.4, .6] # Size of obstacles
param.q = 1e2 # Reaching task weight term
param.q2 = 1e0 # Obstacle avoidance task weight term
param.r = 1e-3 # Control weight term

# Task parameters
# ===============================

# Target
param.Mu = np.array([
    [.9], [.9], [np.pi/6]
    ]) # Viapoint [x1,x2,o]

param.A = np.zeros((2, 2, param.nbPoints))
for t in range(param.nbPoints):
    param.A[:, :, t] = np.asarray([
        [np.cos(param.Mu[2, t]), -np.sin(param.Mu[2, t])], 
        [np.sin(param.Mu[2, t]), np.cos(param.Mu[2, t])], 
    ])
    
Q = np.identity(param.nbVarX * param.nbPoints) * param.q #Precision matrix to reach viapoints
R = np.identity((param.nbData-1) * param.nbVarU) * param.r #control weight matrix (at trajectory level)

tl = np.linspace(0, param.nbData, param.nbPoints+1)
tl = np.rint(tl[1:]).astype(np.int64)-1
idx = np.arange(0, param.nbVarX, step=1) + np.array(tl* param.nbVarX)

# GPIS representation of obstacles
param.p = np.asarray([1.4, 1e-5])
param.x = np.asarray([[0.2, 0.4, 0.6, -0.4, 0.6, 0.9], 
           [0.5, 0.5, 0.5, 0.8, 0.1, 0.6]])
param.y = np.asarray([[-1, 0, 1, -1, -1, -1],])

# Disc as geometric prior
rc = 4e-1
xc = np.asarray([[.05], [.05]]) + .5 # location of disc
S = np.identity(2) * np.power(rc, -2)

param.Mu2 = .5 * rc * np.diag(1 - (param.x - np.tile(xc,(1,param.x.shape[1]))).T @ S @ (param.x - np.tile(xc,(1,param.x.shape[1])))).reshape((1, -1))
param.K, _ = covFct(param.x, param.x, param.p, 1) # Inclusion of noise on the inputs for the computation of K

# Iterative LQR (iLQR)
u = np.zeros((param.nbVarU * (param.nbData-1), 1))
x0 = np.asarray([[0.3], [0.05]])

# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([np.zeros((param.nbVarX, param.nbVarX*(param.nbData-1))), 
np.tril(np.kron(np.ones((param.nbData-1, param.nbData-1)), np.eye(param.nbVarX)*param.dt))]) 
Sx0 = np.kron(np.ones([param.nbData, 1]) , np.identity(param.nbVarX) )
Su = np.zeros((param.nbVarX, Su0.shape[1])) # We remove the lines that are out of interest
for i in range(param.nbVarX):
    Su[i, :] = Su0[int(idx[i]), :] 

# Solving iLQR
# ===============================

for i in range( param.nbIter ):
    x = Su0 @ u + Sx0 @ x0
    x = x.reshape((param.nbData, param.nbVarX)).T
   
    f, J = f_reach(x[:, tl.flatten()], param)  # Tracking objective
    f2, J2, id2 , _ = f_avoid(x, param)# Avoidance objective
    
    if id2.size > 0: # Numpy does not allow zero sized array as Indices
        Su2 = Su0[id2.flatten()]
        du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + Su2.T @ J2.T @ J2 @ Su2 * param.q2 + R) @ \
                (-Su.T @ J.T @ Q @ f - Su2.T @ J2.T @ f2 * param.q2 - u * param.r)
    else: # It means that we have a collision free path
        du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + R) @ \
                (-Su.T @ J.T @ Q @ f - u * param.r)

    # Perform line search
    alpha = 1
    cost0 = np.linalg.norm(f)**2 * param.q + np.linalg.norm(f2)**2 * param.q2 + np.linalg.norm(u) * param.r

    while True:
        utmp = u + du * alpha
        xtmp = Su0 @ utmp + Sx0 @ x0
        xtmp = xtmp.reshape((param.nbData, param.nbVarX)).T
        ftmp, _ = f_reach(xtmp[:, tl.flatten()], param)
        f2tmp,_,_,_ = f_avoid(xtmp, param)
        cost = np.linalg.norm(ftmp)**2 * param.q + np.linalg.norm(f2tmp)**2 * param.q2 + np.linalg.norm(utmp) * param.r

        if cost < cost0 or alpha < 1e-3:
            u = utmp
            print("Iteration {}, cost: {}, alpha: {}".format(i,cost,alpha))
            break

        alpha /=2
    
    if np.linalg.norm(alpha * du) < 1e-2: # Early stop condition
        break

# Plots
# ===============================

limAxes = [0, 1, 0, 1]
nbRes = 30

xx, yy = np.meshgrid(np.linspace(limAxes[0], limAxes[1], nbRes), 
                     np.linspace(limAxes[2], limAxes[3], nbRes))

xtmp = np.array([xx.ravel(), yy.ravel()])
msh = np.array([[np.min(xtmp[0,:]), np.min(xtmp[0,:]), np.max(xtmp[0,:]), np.max(xtmp[0,:]), np.min(xtmp[0,:])],
                [np.min(xtmp[1,:]), np.max(xtmp[1,:]), np.max(xtmp[1,:]), np.min(xtmp[1,:]), np.min(xtmp[1,:])]])

# Avoidance
f2, _, _, idt = f_avoid(xtmp, param)
z2 = np.zeros((nbRes**2, 1))
z2[idt] = f2**2
zz2 = np.reshape(z2, (nbRes, nbRes))

fig, ax = plt.subplots()
ax.contourf(xx, yy, zz2-np.max(zz2), cmap="gray_r")
ax.plot(msh[0,:], msh[1,:], linewidth=1, color=[0, 0, 0], zorder=9) # border
ax.plot(param.Mu[0,:], param.Mu[1,:], marker='o', markersize=10, markeredgecolor=[.8, 0, 0], markerfacecolor=[.8, 0, 0], zorder=10) # viapoints
ax.plot(x[0,[0,-1]], x[1,[0,-1]], linewidth=2, color=[.7, .7, .7], zorder=9) # initialization
ax.plot(x[0,:], x[1,:], linewidth=2, color=[0, 0, 0], zorder=9)
if len(tl) > 1:
    ax.plot(x[0,[0,tl[:-1]]], x[1,[0,tl[:-1]]], marker='o', markersize=10, markeredgecolor=[0, 0, 0], markerfacecolor=[0, 0, 0], zorder=10)
else:
    ax.plot(x[0, 0], x[1, 0], marker='o', markersize=10, markeredgecolor=[0, 0, 0], markerfacecolor=[0, 0, 0], zorder=10)
ax.set_xlim(limAxes[0], limAxes[1])
ax.set_ylim(limAxes[2], limAxes[3])
ax.axis('off')

plt.show()
