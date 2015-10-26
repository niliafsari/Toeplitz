#!/usr/bin/env python

import sys
import numpy as np
from scipy.linalg import inv, toeplitz
from toeplitz_decomp import *
#from new_parallel import *
from parallel_gather_decomposition import *
from mpi4py import MPI

# 1-d case
a=np.load('simdata.npy')
A=np.copy(toeplitz(a[:,1]))
A_orig=np.copy(toeplitz(a[:,1]))
b=1
pad=2
l=block_toeplitz_par(A,1,0)
result = np.dot(np.conj(l).T,l)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank==0:
	#n=A.shape[1]/b 
	#n=n+n*pad
	#k=l[:,n-1]
	#k=np.reshape(k,(n,1))
	#print np.sum(l[0:n-2,n-2]-l[1:n-1,n-1])\
	print("Consistency check, these numbers should be small:",np.sum(A_orig-result[0:512,0:512]))
	

if rank==0:
	ll=toeplitz_blockschur(np.conj(a[:,1:2].T),1,0)
	result = np.dot(np.conj(ll).T,ll)
	print("Consistency check, these numbers should be small:",np.sum(A_orig-result[0:512,0:512]))