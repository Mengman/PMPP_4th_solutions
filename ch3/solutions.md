1. In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.

a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.

b. Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design.

c. Analyze the pros and cons of each of the two kernel designs

Kernel A is a Row-wise kernel

**Pros**:
* Better coalesced memory access for matrix M and P, which can lead to improved memory bandwidth utilization.
* Potentially better cache utilization for matrix M, as each thread reuses the same row of A for all computations.

**Cons**:
* Strided memory access for matrix **j**, which can lead to reduced memory bandwidth utilization.
* May not perform well for very wide matrices (large **j**) due to increased register pressure and potential lack of parallelism for small **i**.

Kernel B is a Column-wise kernel

**Pros**:
* Better coalesced memory access for matrix N and P, which can lead to improved memory bandwidth utilization.
* Potentially better cache utilization for matrix N, as each thread reuses the same column of N for all computations.

**Cons**:
* Strided memory access for matrix M, which can lead to reduced memory bandwidth utilization.
* May not perform well for very tall matrices (large **i**) due to increased register pressure and potential lack of parallelism for small **j**.

General comparison:

The performance of these kernels can vary depending on the dimensions of the input matrices (i, j, and k).

The row-wise kernel might perform better when i is much larger than j, while the column-wise kernel might be preferable when j is much larger than i.

Both kernels have room for optimization, such as using shared memory to improve data reuse and reduce global memory accesses.

Neither kernel utilizes the full potential of the GPU's parallelism, as they don't exploit the 2D nature of thread blocks. A more efficient approach would be to use 2D thread blocks to compute tiles of the output matrix.

<hr/>


