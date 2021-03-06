import numpy as np
from mpi4py import MPI
from scipy.linalg import toeplitz
from toeplitz_decomp import *

def block_toeplitz_par(a,b):
  comm = MPI.COMM_WORLD
  size  = comm.Get_size()
  rank = comm.Get_rank()
  n=a.shape[1]/b
  g1=np.zeros(shape=(b,n*b), dtype=complex)
  g2=np.zeros(shape=(b,n*b), dtype=complex)
#for simulated data that the first matrix is toeplitz
#c=toeplitz_decomp(np.array(a[0:b,0]).reshape(-1,).tolist())
  if rank==0:
	  #c=myBlockChol(a[0:b,0:b],1)
	  c=toeplitz_decomp(np.array(a[0:b,0]).reshape(-1,).tolist())
	  for j in xrange(0,n): 
	       g2[:,j*b:(j+1)*b]= -np.dot(inv(c),a[0:b,j*b:(j+1)*b]) 
	       g1[:,j*b:(j+1)*b]= -g2[:,j*b:(j+1)*b]
  g1=comm.bcast(g1 ,root=0)		
  g2=comm.bcast(g2,root=0)
  size_node_temp=(n//size)*b
  size_node=size_node_temp
  if rank==size-1:
	size_node = (n//size)*b + (n%size)*b
  start = rank*size_node_temp
  end = min(start+size_node, n*b)
  l=np.zeros(shape=(n*b,n*b), dtype=complex)
  empty=0
  if rank==0:
  	l[0:b,:] = g1
  for i in xrange( 1, n):    
    global_start_g1=0
    global_end_g1=(n-i)*b
    global_start_g2=i*b
    global_end_g2=n*b
    start_g1=start
    if (global_end_g1<end and start<global_end_g1):
    	end_g1=min(end,global_end_g1)
    elif (global_end_g1<end and start>=global_end_g1):
        empty=1
        end_g1=start
    else: 
	end_g1=end
    start_g2=start_g1+global_start_g2
    end_g2=end_g1+global_start_g2
    g2[:,0:start_g2]=0
    g2[:,end_g2:n*b]=0
    for j in xrange(0,b):
	if rank==0:	
		g0_1=np.copy(g1[j,start_g1+j])
		g0_2=np.copy(g2[:,start_g2+j])
	else:
		g0_1=np.zeros(shape=(1,1), dtype=complex)
		g0_2=np.zeros(shape=(b,1), dtype=complex)
	g0_1=comm.bcast(g0_1 ,root=0)		
	g0_2=comm.bcast(g0_2,root=0)
	if empty:
		continue
	if g0_2.all()==0:
		g2[:,start_g2:end_g2]=-g2[:,start_g2:end_g2]
		continue		
	sigma=np.dot(np.conj(g0_2.T),g0_2)
   alpha=-np.sign(g0_1)*np.sqrt(g0_1**2.0 - sigma)
	z=g0_1+alpha
	x2=-np.copy(g0_2)/np.conj(z)
	beta=(2*z*np.conj(z))/(np.conj(z)*z-sigma)
	if rank==0 :
		g1[j,start_g1+j]=-alpha
		g2[:,start_g2+j]=0
		v=np.copy(g1[j,start_g1+j+1:end_g1]+np.dot(np.conj(x2.T),g2[:,start_g2+j+1:end_g2]))
		g1[j,start_g1+j+1:end_g1]=g1[j,start_g1+j+1:end_g1]-beta*v
		v=np.reshape(v,(1,v.shape[0]))
		x2=np.reshape(x2,(1,x2.shape[0]))
		g2[:,start_g2+j+1:end_g2]=-g2[:,start_g2+j+1:end_g2]-beta*np.dot(x2.T,v)
	else:
		v=np.copy(g1[j,start_g1:end_g1]+np.dot(np.conj(x2.T),g2[:,start_g2:end_g2]))
		g1[j,start_g1:end_g1]=g1[j,start_g1:end_g1]-beta*v
		v=np.reshape(v,(1,v.shape[0]))
		x2=np.reshape(x2,(1,x2.shape[0]))
		g2[:,start_g2:end_g2]=-g2[:,start_g2:end_g2]-beta*np.dot(x2.T,v)
    G=np.zeros_like(g2)
    comm.Allreduce(g2, G, op=MPI.SUM)
    g2=G
    if (empty==0):			
	    l[i*b: (i+1)*b,i*b+start_g1:i*b+end_g1]=g1[:,start_g1:end_g1]
  L=np.zeros_like(l)
  comm.Allreduce(l, L, op=MPI.SUM)
  return L
