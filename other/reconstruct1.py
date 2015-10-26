import sys
import numpy as np
from scipy.linalg import inv, toeplitz, hankel
from numpy import linalg as LA
#from mpi4py import MPI
#import h5py 
import time
from sp import multirate
from toeplitz_decomp import *
import matplotlib.pyplot as plt
from scipy import signal


def reconstruct_map(uc,lr,meff_f,pad):
	if pad:
		n=uc.shape[0]/(meff_f*(pad+1))
	else:
		print "one round of padding necessary" 
		return 0
	m=meff_f/4
	result=np.zeros(shape=(n*2,meff_f),dtype=complex)
	offset=0
	for i in xrange(0,2*n):
		if i<=(n-1):
			result[n-1-i,0:meff_f]=np.fliplr(np.roll(uc[(pad*uc.shape[0]/(pad+1))+i*(meff_f)+offset:(pad*uc.shape[0]/(pad+1))+i*(meff_f)+offset+meff_f].T,2*m))
		elif i>=n:
			result[i,0:meff_f]=np.roll(lr[(pad*lr.shape[0]/(pad+1))+(i-n-1)*(meff_f)+offset:(pad*lr.shape[0]/(pad+1))+(i-n-1)*(meff_f)+offset+meff_f].T,2*m+1)
	return result
