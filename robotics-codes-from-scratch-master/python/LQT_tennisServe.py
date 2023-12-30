'''
Linear Quadratic tracker applied on a via point example

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Yiming Li <yiming.li@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt

def transferMatrices(A, B):
    nbVax, nbVarU, nbData = B.shape
    nbData += 1
    Sx = np.kron(np.ones((nbData,1)), np.eye(nbVax))
    Su = np.zeros((nbVax*(nbData), nbVarU*(nbData-1)))
    for t in range(nbData-1):
        id1 = t*nbVax
        id2 = (t+1)*nbVax
        id3 = (t+2)*nbVax
        id4 = t*nbVarU
        id5 = (t+1)*nbVarU
        Sx[id2:id3,:] = A[:,:,t] @ Sx[id1:id2,:]
        Su[id2:id3,:] = A[:,:,t] @ Su[id1:id2,:]
        Su[id2:id3,id4:id5] = B[:,:,t]
    return Su, Sx

# Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.nbData = 200  # Number of datapoints
param.nbAgents = 3 #Number of agents (left hand, right hand, ball)
param.nbVarPos = 2 * param.nbAgents # Dimension of position data (here: x1,x2 for the three agents)
param.nbDeriv = 2 # number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
param.nbVar = param.nbVarPos * (param.nbDeriv+1) # Dimension of state vector (position, velocity and force)
param.nbVarU = 4 # Number of control variables (acceleration commands for the two hands)
param.dt = 1e-2  # Time step length
param.rfactor = 1e-8
param.R = np.eye((param.nbData-1) * param.nbVarU) * param.rfactor  # Control cost matrix
m = [0.2, 0.2, 0.2] # Mass of Agents (left hand, right hand and ball)
g = np.array([0, -9.81]) # Gravity vector
tEvent = np.array([50, 100, 150]) # Time stamps when the ball is released, when the ball is hit, and when the hands come back to their initial pose
x01 = np.array([1.6, 0]) # Initial position of Agent 1 (left hand)
x02 = np.array([2, 0]) # Initial position of Agent 2 (right hand)
xTar = np.array([1, -0.2]) # Desired target for Agent 3 (ball)

# Linear dynamical system parameters
Ac1 = np.kron(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), np.eye(2))
Bc1 = np.kron(np.array([0, 1, 0]), np.eye(2))
Ac = np.kron(np.eye(param.nbAgents), Ac1)
Bc = np.concatenate((np.kron(np.eye(2), Bc1).T, np.zeros((6, param.nbVarU))), axis=0) # Ball is not directly controlled

# Parameters for discrete dynamical system
Ad = np.eye(param.nbVar) + Ac * param.dt
Bd = Bc * param.dt

# Initialize A and B
A = np.tile(Ad[:,:,np.newaxis], [1, 1, param.nbData-1])
B = np.tile(Bd[:,:,np.newaxis], [1, 1, param.nbData-1])
# Set Agent 3 state (ball) equals to Agent 1 state (left hand) until tEvent(1)
A[12:16, :, :tEvent[0]] = 0
A[12:14, :2, :tEvent[0]] = np.tile(np.eye(2)[:,:,np.newaxis], (1, 1, tEvent[0]))
A[12:14, 2:4, :tEvent[0]] = np.tile(np.eye(2)[:,:,np.newaxis], (1, 1, tEvent[0])) * param.dt
A[14:16, 2:4, :tEvent[0]] = np.tile(np.eye(2)[:,:,np.newaxis], (1, 1, tEvent[0]))
A[14:16, 4:6, :tEvent[0]] = np.tile(np.eye(2)[:,:,np.newaxis], (1, 1, tEvent[0])) * param.dt

# Set Agent 3 state (ball) equals to Agent 2 state (right hand) at tEvent(2)
A[12:16, :, tEvent[1]-1] = 0
A[12:14, 6:8, tEvent[1]-1] = np.eye(2)
A[12:14, 8:10, tEvent[1]-1] = np.eye(2) * param.dt
A[14:16, 8:10, tEvent[1]-1] = np.eye(2)
A[14:16, 10:12, tEvent[1]-1] = np.eye(2) * param.dt

print(A)

# Build transfer matrices
Su, Sx = transferMatrices(A, B)

# Task setting
Mu = np.zeros((param.nbVar*param.nbData,1)) #Sparse reference
Q = np.zeros((param.nbVar*param.nbData, param.nbVar*param.nbData)) #Sparse precision matrix

# Agent 2 and Agent 3 must meet at tEvent(2) (right hand hitting the ball)
id = np.array([6, 7, 12, 13]) + (tEvent[1]-1)*param.nbVar
Q[np.ix_(id,id)] = np.eye(4)
Q[id[0]:id[1]+1, id[2]:id[3]+1] = -np.eye(2) #Common meeting point for the two agents
Q[id[2]:id[3]+1, id[0]:id[1]+1] = -np.eye(2) #Common meeting point for the two agents

# Agent 1 (left hand) and Agent 2 (right hand) must come back to initial pose at tEvent(3) and stay here
for t in range(tEvent[2]-1, param.nbData):
    id = np.arange(4) + (t-1)*param.nbVar #Left hand
    Q[np.ix_(id,id)] = np.eye(4) * 1e3
    Mu[id] = np.concatenate((x01[:,np.newaxis], np.zeros((2,1))))
    id = np.arange(4) + (t-1)*param.nbVar + 6 #Right hand
    Q[np.ix_(id,id)] = np.eye(4) * 1e3
    Mu[id] = np.concatenate((x02[:,np.newaxis], np.zeros((2,1))))

# Agent 3 (ball) must reach desired target at the end of the movement
id = np.array([12, 13]) + (param.nbData-1)*param.nbVar
Q[np.ix_(id,id)] = np.eye(2)
Mu[id] = xTar[:,np.newaxis]

# Problem solved with linear quadratic tracking (LQT)
x0 = np.concatenate((x01, np.zeros(2), m[0]*g, x02, np.zeros(2), m[1]*g, x01, np.zeros(2), m[2]*g))[:,np.newaxis] #Initial state
u = np.linalg.pinv(Su.T @ Q @ Su + param.R) @ Su.T @ Q @ (Mu - Sx @ x0) # Estimated control commands
x = (Sx @ x0 + Su @ u).reshape((param.nbData,param.nbVar)).T # Generated trajectory

# Plot
fig, ax = plt.subplots(figsize=(12,8))
# Agents
plt.plot(x[0,:], x[1,:], '-', linewidth=4, color='black') # Agent 1 (left hand)
plt.plot(x[6,:], x[7,:], '-', linewidth=4, color='gray') # Agent 2 (right hand)
plt.plot(x[12,:], x[13,:], ':', linewidth=4, color='orange') # Agent 3 (ball)

# Events
plt.plot(x[0,0], x[1,0], '.', markersize=40, color='black') # Initial position (left hand)
plt.plot(x[6,0], x[7,0], '.', markersize=40, color='gray') # Initial position (right hand)
plt.plot(x[0,tEvent[0]-1], x[1,tEvent[0]-1], '.', markersize=40, color='green') # Release of ball
plt.plot(x[6,tEvent[1]-1], x[7,tEvent[1]-1], '.', markersize=40, color='blue') # Hitting of ball
plt.plot(xTar[0], xTar[1], '.', markersize=40, color='red') # Ball target

plt.legend(['Left hand motion (Agent 1)','Right hand motion (Agent 2)','Ball motion (Agent 3)', 'Left hand initial point', 'Right hand initial point','Ball releasing point','Ball hitting point','Ball target'], fontsize=20,loc='upper left')
plt.axis('equal')
plt.show()
