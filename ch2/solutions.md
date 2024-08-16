1. If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?
(A) i=threadIdx.x + threadIdx.y;
(B) i=blockIdx.x + threadIdx.x;
(C) i=blockIdx.x * blockDim.x + threadIdx.x;
(D) i=blockIdx.x * threadIdx.x;

    **Correct answer: C**
```
For one dimension thread block, threadIdx.x is the thread id in its block and starts with 0; blockDim.x is the total thread number in one thread block; blockIdx.x is the block id.

We can use global thread id to indices to the data index. So, the global thread id equals blockIdx.x * blockDim.x + threadIdx.x
```

2. Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?
(A) i=blockIdx.x*blockDim.x + threadIdx.x +2;
(B) i=blockIdx.x* threadIdx.x *2;
(C) i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;
(D) i=blockIdx.x * blockDim.x * 2 + threadIdx.x;

    **Correct answer: C**
```
For vector a=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; thread id=0 responsible for vector elements [0, 1], thread id=0 responsible for vector elements [2, 3], thread id=2 responsible for [4, 5], and so on. And global thread id equals (blockIdx.x * blockDim.x + threadIdx.x), the first element index i equals (global thread id) * 2
```

3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes 2*blockDim.x consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?
(A) i=blockIdx.x * blockDim.x + threadIdx.x +2;
(B) i=blockIdx.x * threadIdx.x * 2;
(C) i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;
(D) i=blockIdx.x * blockDim.x * 2 + threadIdx.x;

    **Correct answer: D**
```
Each thread block responsible for (blockDim.x * 2) elements, so every block offset is (blockIdx.x * blockDim.x * 2).
```

4 C
5 D
6 D
7 C
8 C
9a 128
9b 200064
9c 1563
9d 200064
9e 200000
10 declare those functions with both `__host__` and `__device__` key word