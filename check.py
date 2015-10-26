#from numba import jit
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
from reconstruct1 import *
def backwardmap(input_f,n,m):
	result=np.zeros(shape=(n*2,m), dtype=complex)
	

np.set_printoptions(precision=2, suppress=True, linewidth=5000)
if len(sys.argv) < 8:
    print "Usage: %s filename num_rows num_columns offsetn offsetm sizen sizem" % (sys.argv[0])
    sys.exit(1)

num_rows=int(sys.argv[2])
num_columns=int(sys.argv[3])
offsetn=int(sys.argv[4])
offsetm=int(sys.argv[5])
sizen=int(sys.argv[6])
sizem=int(sys.argv[7])

if offsetn>num_rows or offsetm>num_columns or offsetn+sizen>num_rows or offsetm+sizem>num_columns:
	print "Error sizes or offsets don't match"
	sys.exit(1)

a = np.memmap(sys.argv[1], dtype='float32', mode='r', shape=(num_rows,num_columns),order='F')

plt.subplot(1, 1, 1)
plt.imshow(a.T, interpolation='nearest')
plt.colorbar()
plt.show()
if sizem==1:
	pad2=0
	const=0
pad=1
pad2=1
debug=1

neff=sizen+sizen*pad
meff=sizem+sizem*pad
if sizem==1:
	meff=sizem
if sizem==1:
	meff=sizem
a_input=np.zeros(shape=(neff,meff), dtype=complex)
a_input[:sizen,:sizem]=np.copy(a[offsetn:offsetn+sizen,offsetm:offsetm+sizem])
del a

a_input=np.where(a_input > 0, a_input, 0)
const=meff/2
a_input=np.sqrt(a_input)
if debug:
	print a_input,"after sqrt"

a_input[:sizen,:sizem]=np.fft.fft2(a_input,s=(sizen,sizem))

if debug:
	print a_input,"after first fft"

if sizem>1:
	a_input[neff-(sizen/2-1):neff,0:sizem/2]=a_input[sizen/2+1:sizen,0:sizem/2]
	a_input[0:sizen/2,meff-(sizem/2-1):meff]=a_input[0:sizen/2,sizem/2+1:sizem]
	a_input[neff-(sizen/2-1):neff,meff-(sizem/2-1):meff]=a_input[sizen/2+1:sizen,1+sizem/2:sizem]
	a_input[sizen/2:sizen,:sizem]=np.zeros(shape=(sizen/2,sizem))
	a_input[:sizen/2,sizem/2:sizem]=np.zeros(shape=(sizen/2,sizem/2))
else:
	a_input[neff-(sizen/2-1):neff,0]=a_input[sizen/2+1:sizen,0]
	a_input[sizen/2:sizen,0:1]=np.zeros(shape=(sizen/2,1))

print a_input
if debug:
	print a_input/np.sqrt(neff*meff),"after shift"
cj=a_input/np.sqrt(neff*meff)


#corr=np.zeros(shape=(neff, meff), dtype=complex)
#for i in xrange(0,neff):
#	for j in xrange(0,meff):		
#		temp = np.roll(a_input,j,axis=1)
#		corr[i,j]=signal.correlate(a_input, np.roll(temp,i,axis=0),mode='valid')[0,0] /(neff*meff+0j)	
#print corr,"corr"

a_input=np.fft.ifft2(a_input,s=(neff,meff))
if debug:
	print a_input,"after second fft"
a_input=np.power(np.abs(a_input),2)
if debug:
	print a_input,"after abs^2"
a_input=np.fft.fft2(a_input,s=(neff,meff)) 
if debug:
	print a_input,"after third fft"
	

if debug:
	a_inp=np.fft.fft2(a_input,s=(neff,meff)) 
	a_inp=np.sort(np.reshape(a_inp,(1,neff*meff)))
	print a_inp,"check"
meff_f=meff+pad2*meff
epsilon=np.identity(meff_f)*10e-8
if sizem==1:
	const=0
	meff_f=1
	meff=1
	pad2=0
input_f=np.zeros(shape=(neff*meff_f, neff*meff_f), dtype=complex)

print neff
for i in xrange(0,neff):
	for j in xrange(i,neff):
		if (j-i)<neff/2 and j>i:
			rows=np.append(a_input[j-i,:meff-const],np.zeros(pad2*meff+const))
			cols=np.append(np.append(a_input[j-i,0],a_input[j-i,const+1:][::-1]),np.zeros(pad2*meff+const))
			input_f[i*meff_f:(i+1)*meff_f,j*meff_f:(j+1)*meff_f]=toeplitz(cols,rows)
		elif j==i:
			#print a_input[j-i,:meff-const].shape,np.zeros(pad2*meff+const).shape,meff_f
			input_f[i*meff_f:(i+1)*meff_f,j*meff_f:(j+1)*meff_f]=toeplitz(np.conj(np.append(a_input[j-i,:meff-const],np.zeros(pad2*meff+const))))

input_f=np.conj(np.triu(input_f).T)+np.triu(input_f,1)
#print input_f,"input_f"

#print input_f[:,neff*meff_f-1].T,"last column"

	#print lr.shape
#print ur.shape,input_f[:,neff*meff_f-1:neff*meff_f].shape,input_f.shape,meff_f

#print np.concatenate((lr.T,input_f[:,neff*meff_f-1:neff*meff_f]),axis=1)[384/2:384,:]



#for i in xrange(0,neff/2):
#	print input_f[(i+1)*meff_f-1,(neff/2-1)*meff_f:(neff/2)*meff_f].shape
#	print input_f[(i+1)*meff_f-1,(neff/2-1)*meff_f:(neff/2)*meff_f]
#print neff, meff_f
#if debug:
	#print input_f[0:meff+pad2*meff,0:(meff)], "blocks00"
#	print input_f[0:meff+pad2*meff,meff+pad2*meff:meff_f+(meff)],"blocks01"
	#print input_f[meff+pad2*meff:meff_f+(meff),0:meff+pad2*meff],"blocks10"
#	print input_f[0:meff+pad2*meff,2*meff_f:2*meff_f+meff_f],"blocks02"
#	print input_f[0:meff+pad2*meff,3*meff_f:3*meff_f+(meff)],"blocks03"
#	print input_f[0:meff+pad2*meff,4*meff_f:4*meff_f+(meff_f)],"blocks04"
#print input_f[0:meff+pad2*meff,5*meff_f:5*meff_f+(meff)],"blocks05"
#print input_f[0:8,16:32], neff,meff_f
#print np.sum(input_f, axis=1),"sums"

#input_f=epsilon+input_f
#w, v = LA.eig(input_f)
#print w,"values"
#print v,"vectors"
#print np.sort(np.real(np.linalg.eigvals(input_f))),"eig"
#input_f=input_f+epsilon
#L=np.linalg.cholesky(input_f)
#L=np.conj(L.T)

#output_file="processedData/gate0_numblock_%s_meff_%s_offsetn_%s_offsetm_%s.dat" %(str(sizen),str(meff_f),str(offsetn),str(offsetm))
#output = np.memmap(output_file, dtype='complex', mode='w+', shape=(meff_f, sizen*meff_f),order='F')
#output[:,:]=input_f[:meff_f,:]
#del output
#print input_f[0:sizen*meff_f, sizem*meff_f:neff*meff_f]
ur=np.zeros(shape=(1,0))
lc=np.zeros(shape=(1,0))
if debug:
	pad=5
	u=toeplitz_blockschur(input_f[:neff*meff_f/2,:neff*meff_f/2],meff_f,pad)
	#print u
	l=np.conj(u.T)
	t=np.dot(l,u)
	uc=t[:,(neff/2)*(pad+1)*(meff_f)-meff_f/2-1:(neff/2)*(pad+1)*(meff_f)-meff_f/2]
	lr=t[(neff/2)*(pad+1)*(meff_f)-meff_f/2-1:(neff/2)*(pad+1)*(meff_f)-meff_f/2,:].T
	results=reconstruct_map(uc,lr,meff_f,pad)
#if debug:
#	for i in xrange(0,(neff/2)*(pad+1)):
#		print u[(i)*meff_f:(i+1)*meff_f,(neff/2)*(pad+1)*(meff_f)-meff_f:(neff/2)*(pad+1)*meff_f], str(i)

#print np.sum(np.power(np.abs(x),2))



print results,"reconstrcuted Toeplitz"

uc=u[:,(neff/2)*(pad+1)*(meff_f)-meff_f/2-1:(neff/2)*(pad+1)*(meff_f)-meff_f/2]
lr=np.zeros_like(uc)
results=reconstruct_map(uc,lr,meff_f,pad)
print results,"reconstrcuted conjugate"


print np.sum(np.power(np.abs(uc),2)),np.sum(np.power(np.abs(ur),2)), a_input[0,0]
corr=np.zeros(shape=(neff, meff_f), dtype=complex)
for i in xrange(0,neff):
	for j in xrange(0,meff_f):		
		temp = np.roll(results,j,axis=1)
		corr[i,j]=signal.correlate(results,np.roll(temp,i,axis=0),mode='valid')[0,0] 
print corr,"corr"