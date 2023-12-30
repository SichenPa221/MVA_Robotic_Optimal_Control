"""
Concatenated Bernstein basis functions with constraints to encode
a signed distance function (2D inputs, 1D output)

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Yiming Li <yiming.li@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
"""

import numpy as np
#from scipy.linalg import block_diag
import matplotlib.pyplot as plt

def binomial(n, i):
    if n >= 0 and i >= 0:
        b = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
    else:
        b = 0
    return b

def block_diag(A,B):
    out = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]))
    out[:A.shape[0], :A.shape[1]] = A
    out[A.shape[0]:, A.shape[1]:] = B
    return out

# Parameters
nbFct = 4  # Number of basis functions for each dimension
nbSeg = 5  # Number of segments for each dimension
nbIn = 2  # Dimension of input data (here: 2D surface embedded in 3D)
nbOut = 1  # Dimension of output data (here: height data)
nbDim = 40  # Grid size for each dimension

# Reference surface
data = np.load('../data/sdf01.npy',allow_pickle='True').item()
x0 =  data['y'].T

# Input array
T1, T2 = np.meshgrid(np.linspace(0, 1, nbDim), np.linspace(0, 1, nbDim))
t12 = np.vstack((T1.flatten(), T2.flatten()))

# Bézier curve in matrix form
nbT = nbDim // nbSeg
t = np.linspace(0, 1 - 1 / nbT, nbT)
T0 = np.zeros((len(t), nbFct))
for n in range(nbFct):
    T0[:, n] = np.power(t, n)

B0 = np.zeros((nbFct, nbFct))
for n in range(1, nbFct + 1):
    for i in range(1, nbFct + 1):
        B0[nbFct - i, n - 1] = (-1) ** (nbFct - i - n) * (-binomial(nbFct - 1, i - 1)) * binomial(nbFct - 1 - (i - 1), nbFct - 1 - (n - 1) - (i - 1))
T = np.kron(np.eye(nbSeg), T0)
B = np.kron(np.eye(nbSeg), B0)

C0 = np.array([[1, 0, 0, -1], [0, 1, 1, 2]]).T
C0 = block_diag(np.eye(nbFct-4),C0)

C = np.eye(2)
for n in range(nbSeg-1):
    C = block_diag(C,C0)
C = block_diag(C,np.eye(nbFct-2))

phi = T @ B @ C
Psi = np.kron(phi, phi)

# Encoding and reproduction with basis functions
wb = np.linalg.pinv(Psi) @ x0
xb = Psi @ wb

# Plot
fig = plt.figure(figsize=(16, 8))

# Reference surface
ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('off')
# ax1.plot_surface(T1, T2, np.reshape(x0, (nbDim, nbDim)) - np.max(x0), cmap='viridis', edgecolor='k')
ax1.contour(T1, T2, np.reshape(x0, (nbDim, nbDim)), levels=np.arange(0, 1, 0.02), linewidths=2)
msh = ax1.contour(T1, T2, np.reshape(x0, (nbDim, nbDim)), levels=[0], linewidths=4, colors='b')
ax1.axis('tight')
ax1.axis('equal')

# Reconstructed surface
ax2 = fig.add_subplot(1, 2, 2)
ax2.axis('off')
# ax2.plot_surface(T1, T2, np.reshape(xb, (nbDim, nbDim)) - np.max(xb), cmap='viridis', edgecolor='k')
ax2.contour(T1, T2, np.reshape(xb, (nbDim, nbDim)), levels=np.arange(0, 1, 0.02), linewidths=2)
msh = ax2.contour(T1, T2, np.reshape(xb, (nbDim, nbDim)), levels=[0], linewidths=4, colors='b')
ax2.axis('tight')
ax2.axis('equal')

# Grid of control points
ttmp = np.linspace(0, 1, int(np.sqrt(len(wb))) + nbSeg - 1)
T1tmp, T2tmp = np.meshgrid(ttmp, ttmp)
ax2.plot(T1tmp, T2tmp, '.', markersize=6, color=[.3, .3, .3])

# Grid of patches
ttmp = np.linspace(0, 1, nbSeg + 1)
T1tmp, T2tmp = np.meshgrid(ttmp, ttmp)
ax2.plot(T1tmp, T2tmp, '.', markersize=12, color=[0, 0, 0])

plt.show()

