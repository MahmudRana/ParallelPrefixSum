# ParallelPrefixSum

This is the implementation of the algorithm Parallel Prefix Sum. The link I have followed is here : https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf
This is a well documented link, published from CMU. I have used mpi4py in python for the implementation. The implementation code is straightforward to understand. I have used Scatter-Gather method for communication between the Processor cores. I solved the parallel prefix sum for N=2^n elements. You need to change the number N in the code to the amount of elements you want to add. Implementing the code to work on any N will be some extra changes. This can be considered as a future work.

* For running the code, you just need to run : https://stackoverflow.com/questions/32257375/how-to-run-a-basic-mpi4py-code
