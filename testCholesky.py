
import scipy.linalg.lapack as ll

import gpu_library as gpu
import numpy as np
import timeit

potrf, potrs = ll.get_lapack_funcs(('potrf', 'potrs'))
'''def _solve_cholesky(Q, z):
     L, info = potrf(Q, lower=False, overwrite_a=False, clean=False)
     if info > 0:
         msg = "%d-th leading minor not positive definite" % info
         raise la.LinAlgError(msg)
     if info < 0:
         msg = 'illegal value in %d-th argument of internal potrf' % -info
         raise ValueError(msg)
     f, info = potrs(L, z, lower=False, overwrite_b=False)
     if info != 0:
         msg = 'illegal value in %d-th argument of internal potrs' % -info
         raise ValueError(msg)
     return f
'''

dimRow = 2000
dimCol = 2000
A = np.random.rand(dimRow, dimCol)
B = np.dot(A, A.transpose())
z = np.random.rand(dimRow)




start_python = timeit.default_timer()
M = gpu.solve_cholesky(np.ravel(B.astype(np.float32), order = 'F'), np.ravel(z.astype(np.float32), order = 'F'), dimRow, 100, 1)
end_python = timeit.default_timer()
print(np.reshape(M, (dimRow, dimCol)))

L, info = potrf(B, lower=False, overwrite_a=False, clean=False)
print(info)
print(L)


