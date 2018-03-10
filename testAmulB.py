import sys
sys.path.append('./build')

import gpu_library
import numpy
import timeit

dimRow = 2000
dimCol = 800
vec1 = numpy.ones((dimRow,dimCol), order = 'F', dtype=numpy.float32)
vec2 = numpy.ones((dimCol,dimRow), order = 'F', dtype=numpy.float32)
start_python = timeit.default_timer()
M = gpu_library.multiply_matrices(vec1, vec2, 3.0, 0.0, dimRow, dimCol, dimCol, dimRow)
end_python = timeit.default_timer()
print("time of A*B Cuda:")
print(end_python - start_python)

start_python = timeit.default_timer()
N = numpy.dot(vec1, vec2)
end_python = timeit.default_timer()
print("time of A*B dans python:")
print(end_python - start_python)

print(M.size)
print("computation finished")
print(M)
