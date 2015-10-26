#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp import *
from new_parallel import *
#from parallel_gather_decomposition import *
from mpi4py import MPI
import scipy.io
#This is a simple blocked Toeplitz test. Size of the matrix: 4*4, size of each block: 2*2

#The test matrix must be blocked positive definite

#the real part of the test matrix
a=np.matrix([[1.000000000000000e+02, -1.100000000000000e+01, 5.200000000000000e+01, -4.300000000000000e+01, 1.800000000000000e+01, -4.800000000000000e+01, 2.000000000000000e+00, -1.500000000000000e+01],
       [-1.100000000000000e+01, 1.950000000000000e+02, -2.000000000000000e+00, 6.900000000000000e+01, 4.800000000000000e+01, 1.000000000000000e+01, 2.000000000000000e+01, 1.500000000000000e+01],
       [5.200000000000000e+01, -2.000000000000000e+00, 1.000000000000000e+02, -1.100000000000000e+01, 5.200000000000000e+01, -4.300000000000000e+01, 1.800000000000000e+01, -4.800000000000000e+01],
       [-4.300000000000000e+01, 6.900000000000000e+01, -1.100000000000000e+01, 1.950000000000000e+02, -2.000000000000000e+00, 6.900000000000000e+01, 4.800000000000000e+01, 1.000000000000000e+01],
       [1.800000000000000e+01, 4.800000000000000e+01, 5.200000000000000e+01, -2.000000000000000e+00, 1.000000000000000e+02, -1.100000000000000e+01, 5.200000000000000e+01, -4.300000000000000e+01],
       [-4.800000000000000e+01, 1.000000000000000e+01, -4.300000000000000e+01, 6.900000000000000e+01, -1.100000000000000e+01, 1.950000000000000e+02, -2.000000000000000e+00, 6.900000000000000e+01],
       [2.000000000000000e+00, 2.000000000000000e+01, 1.800000000000000e+01, 4.800000000000000e+01, 5.200000000000000e+01, -2.000000000000000e+00, 1.000000000000000e+02, -1.100000000000000e+01],
       [-1.500000000000000e+01, 1.500000000000000e+01, -4.800000000000000e+01, 1.000000000000000e+01, -4.300000000000000e+01, 6.900000000000000e+01, -1.100000000000000e+01, 1.950000000000000e+02]])

#the imaginary part
a=a+1.0j*np.matrix([[-0.000000000000000e+00, -1.000000000000000e+00, 4.000000000000000e+00, 2.300000000000000e+01, -3.000000000000000e+00, 1.000000000000000e+00, 2.000000000000000e+00, -3.000000000000000e+00],
       [1.000000000000000e+00, -0.000000000000000e+00, 2.600000000000000e+01, 1.200000000000000e+01, 2.100000000000000e+01, -2.400000000000000e+01, 2.400000000000000e+01, -1.400000000000000e+01],
       [-4.000000000000000e+00, -2.600000000000000e+01, -0.000000000000000e+00, -1.000000000000000e+00, 4.000000000000000e+00, 2.300000000000000e+01, -3.000000000000000e+00, 1.000000000000000e+00],
       [-2.300000000000000e+01, -1.200000000000000e+01, 1.000000000000000e+00, -0.000000000000000e+00, 2.600000000000000e+01, 1.200000000000000e+01, 2.100000000000000e+01, -2.400000000000000e+01],
       [3.000000000000000e+00, -2.100000000000000e+01, -4.000000000000000e+00, -2.600000000000000e+01, -0.000000000000000e+00, -1.000000000000000e+00, 4.000000000000000e+00, 2.300000000000000e+01],
       [-1.000000000000000e+00, 2.400000000000000e+01, -2.300000000000000e+01, -1.200000000000000e+01, 1.000000000000000e+00, -0.000000000000000e+00, 2.600000000000000e+01, 1.200000000000000e+01],
       [-2.000000000000000e+00, -2.400000000000000e+01, 3.000000000000000e+00, -2.100000000000000e+01, -4.000000000000000e+00, -2.600000000000000e+01, -0.000000000000000e+00, -1.000000000000000e+00],
       [3.000000000000000e+00, 1.400000000000000e+01, -1.000000000000000e+00, 2.400000000000000e+01, -2.300000000000000e+01, -1.200000000000000e+01, 1.000000000000000e+00, -0.000000000000000e+00]])

b=2
pad=2
n=a.shape[0]/b
l=np.zeros(shape=(a.shape[0],a.shape[0]), dtype=complex)
z=np.zeros(shape=((n+pad*n)*b,(n+pad*n)*b), dtype=complex)
z[0:n*b,0:n*b]=a
A_orig=np.copy(a)
Z_orig=np.copy(z)
l=toeplitz_blockschur(z,b,0)
#ll=myBlockChol(a,b)
#ll=block_toeplitz_par(a,b,pad)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank==0:
	print l[:,22:24]
	result = np.dot(np.conj(l).T,l)
	#print("Consistency check, these numbers should be small:",np.sum(A_orig-result[0:n*b,0:n*b]))
	print("Consistency check, these numbers should be small:",np.sum(Z_orig-result))
	#np.savetxt("foo.csv", Z_orig-result, delimiter=",")

