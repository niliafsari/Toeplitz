import sys
import numpy as np
from scipy.linalg import toeplitz
#from numpy import linalg as LA
#from toeplitz_decomp import *
#import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import matplotlib.cm as cm
from reconstruct import *
from reconstruct1 import *
import os

if len(sys.argv) < 3:
    print "Usage: %s bina binb" % (sys.argv[0])
    sys.exit(1)

num_rows=16384
num_columns=660
bina=int(sys.argv[2])
binb=int(sys.argv[3])

name='gates/Gates0/'+sys.argv[1]
a1 = np.memmap(sys.argv[1], dtype='float32', mode='r', shape=(num_rows,num_columns),order='F')
name='gates/Gates1/'+sys.argv[1]
a2 = np.memmap(sys.argv[1], dtype='float32', mode='r', shape=(num_rows,num_columns),order='F')
name='gates/Gates6/'+sys.argv[1]
a3 = np.memmap(sys.argv[1], dtype='float32', mode='r', shape=(num_rows,num_columns),order='F')
a=a1+a2+a3/3
del a1
del a2
del a3
a_num=a.shape[0]
b_num=a.shape[1]
#bina=8
#binb=1
a_vi = np.copy(a.reshape(a_num//bina, bina, b_num//binb, binb))

a_view=np.mean(np.mean(a_vi,axis=3),axis=1)
a=a.T
a_view=a_view.T
plt.figure(1)
plt.subplot(211)
plt.imshow(a, aspect='auto', cmap=cm.Greys, interpolation='nearest', vmin=-4, origin='lower')
plt.colorbar()
plt.ylabel("t")
plt.title("dynamic")
plt.xlabel("t")
plt.subplot(212)
plt.imshow(a_view, aspect='auto', cmap=cm.Greys, interpolation='nearest', vmin=-4, origin='lower')
plt.colorbar()
plt.title("dynamic rebined")
plt.ylabel("t")
plt.xlabel("f")
plt.show()
print a_view.shape
output_file=sys.argv[1]+".rebined"
output = np.memmap(output_file, dtype='float32', mode='w+', shape=(a_num//bina,b_num//binb),order='F')
a_view=a_view.T
output[:,:]=a_view[:,:]
del output
del a
