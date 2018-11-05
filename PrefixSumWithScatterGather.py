# I have followed this algorithm
# https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf
# Heartiest thanks to this document maker
import numpy as np
from mpi4py import MPI
import math

# MPI initialization part. Finding the Processor rank, number of Processors, Processor name etc.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
master = 0
status = MPI.Status()

# Number of data you want to work on
N = 16
data = np.zeros(N)
final_result = np.zeros(N)


if (rank == master):
   #Initialize the array in Master. For simplicity we are initializing from 1....N
   data = np.linspace(1, N, N)
   print("Initial Data ", data)


########## Module 1 : Up Sweep Portion ###########
for d in range(0, int(math.log2(N))):
   processor_needed = int(N / int(math.pow(2, d + 1)))
   # Little bit of sequential part, if processor size is more than our need then i don't feel to Parallelize the code
   if(processor_needed<size and rank==master):
       for i in range(0, N-1, int(math.pow(2, d + 1))):
           data[i + int(math.pow(2, d + 1)) - 1] = data[i + int(math.pow(2, d+1)) - 1] + data[i + int(math.pow(2, d)) - 1]
   elif(processor_needed<size and rank!=master):
       pass

   # Else, Parallelize the job
   else:
       for round in range (0, int(processor_needed/size)):
           # Part 1 : Allocate Required Receive buffer Size in each processor and Part of data that will be distributed. It will decrease exponentially. First N, then N/2, then N/2^2.....
           recvbufsize = int(math.pow(2, d + 1))
           recvbuf = np.empty(recvbufsize, dtype='d')
           # Selecting the part of data that need to be scattered
           partdata = np.empty(size * recvbufsize, dtype='d')
           partdata = data[(round*size*recvbufsize):(round*size*recvbufsize)+(size*recvbufsize)]

           # Part 2 : Scatter the Data
           comm.Scatter(partdata, recvbuf, root=0)

           # Part 3 : Change the Data in individual processor after getting the scattered part
           recvbuf[0 + int(math.pow(2, d+1)) - 1] = recvbuf[0 + int(math.pow(2, d + 1)) - 1] + recvbuf[0 + int(math.pow(2, d)) - 1]

           # Part 4 : Gather The data in master
           finalrecvbuf = np.empty(size*recvbufsize, dtype='d')
           comm.Gather(recvbuf, finalrecvbuf, root=0)

           # Part 5 : Finally Arrange the Data
           data[(round * size * recvbufsize):(round * size * recvbufsize) + (size * recvbufsize)] = finalrecvbuf


########### Module 2 : Clearing the final bit ############
final_result[N-1] = data[N-1]
data[N-1] = 0

########### Module 3 : Down Sweep Portion ############
for d in range(int(math.log2(N) - 1), -1, -1):
   processor_needed = int(N / int(math.pow(2, d + 1)))
   # Little bit of sequential part, if processor size is more than our need
   if(processor_needed<size and rank==master):
       for i in range(0, N-1, int(math.pow(2, d + 1))):
           t = data[i+int(math.pow(2,d))-1]
           data[i + int(math.pow(2, d)) - 1] = data[i + int(math.pow(2, d+1)) - 1]
           data[i + int(math.pow(2, d + 1)) - 1] = data[i + int(math.pow(2, d+1)) - 1] + t

   elif(processor_needed<size and rank!=master):
       pass
   # else, parallelize the job
   else:
       for round in range (0, int(processor_needed/size)):
           # Part 1 : Allocate Required Receive buffer Size in each processor and Part of data that will be distributed
           recvbufsize = int(math.pow(2, d + 1))
           recvbuf = np.empty(recvbufsize, dtype='d')
           # Selecting the part of data that need to be scattered
           partdata = np.empty(size * recvbufsize, dtype='d')
           partdata = data[(round*size*recvbufsize):(round*size*recvbufsize)+(size*recvbufsize)]

           # Part 2 : Scatter the Data
           comm.Scatter(partdata, recvbuf, root=0)

           # Part 3 : Change the Data in individual processor after getting the scattered part
           t = recvbuf[0+int(math.pow(2,d))-1]
           recvbuf[0 + int(math.pow(2, d)) - 1] = recvbuf[0 + int(math.pow(2, d+1)) - 1]
           recvbuf[0 + int(math.pow(2, d + 1)) - 1] = recvbuf[0 + int(math.pow(2, d+1)) - 1] + t

           # Part 4 : Gather The data in master
           finalrecvbuf = np.empty(size*recvbufsize, dtype='d')
           comm.Gather(recvbuf, finalrecvbuf, root=0)

           # Part 5 : Finally Arrange the Data
           data[(round * size * recvbufsize):(round * size * recvbufsize) + (size * recvbufsize)] = finalrecvbuf

if (rank == master):
   print("Finished!")
   for i in range (0,N-1):
       final_result[i] = data[i+1]
   print("final data ", final_result)


