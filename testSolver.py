import sys
sys.path.append('./build')

import gpu_library
import numpy as np
import timeit

dimRow = 100000
dimCol = 600
dimColR = 600
A = np.random.rand(dimRow, dimCol)
X = np.random.rand(dimCol, dimColR)
B = np.dot(A, X)
noise = np.random.normal(0, 1, B.shape)
B += noise

start_python = timeit.default_timer()
N = np.linalg.lstsq(A.astype(np.float32), B.astype(np.float32))[0]
end_python = timeit.default_timer()
print("computation time: np.linalg.lstsq")
print(end_python - start_python)
print(np.reshape(N, (dimCol, dimColR)))

M = gpu_library.lin_solver(np.ravel(A.astype(np.float32), order='F'),  np.ravel(B.astype(np.float32), order='F'), dimRow, dimCol, dimColR)

print(np.reshape(M, (dimCol, dimColR)))

print("******************Verifications************************")
print("***B matrix***")
print(B)
print("*** A * X Python ***")
R1 = np.dot(A, N)
print(R1)
print("*** A * X GPU ***")
K = np.reshape(M, (dimCol, dimColR))
R2 = np.dot(A, K)
print(R2)
