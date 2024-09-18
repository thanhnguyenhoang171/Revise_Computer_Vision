import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define points
A = np.array([1, 1, 1])
B = np.array([-1, 1, 1])
C = np.array([1, -1, 1])
D = np.array([-1, -1, 1])
E = np.array([1, 1, -1])
F = np.array([-1, 1, -1])
G = np.array([1, -1, -1])
H = np.array([-1, -1, -1])

camera = np.array([4, 2, 7])

Points = dict(zip("ABCDEFGH", [A, B, C, D, E, F, G, H]))

edges = ["AB", "CD", "EF", "GH", "AC", "BD", "EG", "FH", "AE", "CG", "BF", "DH"]
points = {k: v - camera for k, v in Points.items()}


def pinhole(v):
    x, y, z = v
    if z == 0:
        return np.array([float("inf"), float("inf")])
    return np.array([x / z, y / z])


uvs = {k: pinhole(p) for k, p in points.items()}

plt.figure(figsize=(10, 10))
for a, b in edges:
    ua, va = uvs[a]
    ub, vb = uvs[b]
    plt.plot([ua, ub], [va, vb], "ko-")

plt.axis("equal")
plt.grid()

plt.show()
