import numpy as np


A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

C = A @ B

X = np.array([
    [180, 75, 25],
    [170, 68, 30],
    [190, 90, 27]
])

print("Исходная матрица:")
print(X)

print("\nТранспонированнаяН:")
print(X.T)
