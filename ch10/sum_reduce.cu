__global__ void SimpleSumReductionKernel(float* intput, float* output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}