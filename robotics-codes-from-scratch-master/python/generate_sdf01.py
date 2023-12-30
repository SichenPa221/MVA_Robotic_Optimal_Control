#! /usr/bin/env python3

import numpy as np


def sdCircle(point, center, radius):
    return np.linalg.norm(center - point) - radius


def sdBox(point, center, dimensions):
    d = abs(point - center) - np.array(dimensions) * 0.5
    return np.linalg.norm(np.maximum(d, 0.0)) + min(np.max(d), 0.0)


def smoothUnion(d1, d2, k):
    h = max(k - abs(d1 - d2), 0.0)
    return min(d1, d2) - h * h * 0.25 / k


def sdf(x):
    d = np.ndarray((x.shape[1],))

    for t in range(x.shape[1]):
        xt = x[:, t]
        d[t] = sdCircle(xt, [0.3, 0.36], 0.2)
        d[t] = smoothUnion(d[t], sdBox(xt, [0.5, 0.6], [0.4, 0.2]), 0.1)

    return d



nbDim = 40

t = np.linspace(0, 1, nbDim)

xv, yv = np.meshgrid(t, t)
x = np.array([xv.flatten(), yv.flatten()])

y = sdf(x)


data = {
    'nbDim': nbDim,
    'x': x,
    'y': y,
}

np.save('data/sdf01.npy', data)
