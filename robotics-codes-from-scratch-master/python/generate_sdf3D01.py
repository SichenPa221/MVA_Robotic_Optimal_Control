#! /usr/bin/env python3

import numpy as np


def sdCircle(point, center, radius):
    return np.linalg.norm(center - point) - radius


def sdBox(point, center, dimensions):
    d = abs(center - point) - np.array(dimensions) * 0.5
    return np.linalg.norm(np.maximum(d, 0.0)) + min(np.max(d), 0.0)


def smoothUnion(d1, d2, k):
    h = max(k - abs(d1 - d2), 0.0)
    return min(d1, d2) - h * h * 0.25 / k


def sdf(x):
    d = np.ndarray((x.shape[1],))

    for t in range(x.shape[1]):
        xt = x[:, t]
        d[t] = sdCircle(xt, [0.4, 0.43, 0.5], 0.1)
        d[t] = smoothUnion(d[t], sdBox(xt, [0.5, 0.55, 0.6], [0.2, 0.1, 0.25]), 0.05)

    return d



nbDim = 80

t = np.linspace(0, 1, nbDim)

xv, yv, zv = np.meshgrid(t, t, t)
x = np.array([zv.flatten(), xv.flatten(), yv.flatten()])

y = sdf(x)


data = {
    'nbDim': nbDim,
    'x': x,
    'y': y,
}

np.save('data/sdf3D01.npy', data)
