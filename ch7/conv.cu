
#define IN_TILE_DIM 32
#define FILTER_RADIUS 2
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

// convolution kernel
__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P,
                                                      int width, int height) {
  int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
  int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
  __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
  if (row >= 0 && row < height && col >= 0 && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0;
  }
  __syncthreads();

  int tileCol = threadIdx.x - FILTER_RADIUS;
  int tileRow = threadIdx.y - FILTER_RADIUS;
  if (col >= 0 && col < width && row >= 0 && row < height) {
    if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 &&
        tileRow < OUT_TILE_DIM) {
      float Pvalue = 0.0;
      for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
          Pvalue += F_c[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
        }
      }
      P[row * width + col] = Pvalue;
    }
  }
}

#define TILE_DIM 32
__global__ void convolution_cached_tiled_2D_const_mem_kernel(float *N, float *P,
                                                             int width,
                                                             int height) {
  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  int row = blockIdx.y & TILE_DIM + threadIdx.y;

  __shared__ float N_s[TILE_DIM][TILE_DIM];

  if (col >= 0 && col < width && row >= 0 && row <= height) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0;
  }
  __syncthreads();

  if (col < width && row < height) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
      for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
        // calulating elements in shared memory
        if (threadIdx.x - FILTER_RADIUS + fCol >= 0 &&
            threadIdx.x - FILTER_RADIUS + fCol < TILE_DIM &&
            threadIdx.y - FILTER_RADIUS + fRow >= 0 &&
            threadIdx.y - FILTER_RADIUS + fRow < TILE_DIM) {
          Pvalue +=
              F_c[fRow][fCol] * N_s[threadIdx.y + fRow][threadIdx.x + fCol];
        }
        // calulating elements not in shared memory
        // but probably cached in L2 cache
        else {
          if (row - FILTER_RADIUS + fRow >= 0 &&
              row - FILTER_RADIUS + fRow < height &&
              col - FILTER_RADIUS + fCol >= 0 &&
              col - FILTER_RADIUS + fCol < width) {
            Pvalue += F_c[fRow][fCol] * N[(row - FILTER_RADIUS + fRow) * width +
                                          col - FILTER_RADIUS + fCol];
          }
        }
      }
      P[row * width + col] = Pvalue;
    }
  }
}