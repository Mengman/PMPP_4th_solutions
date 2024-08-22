#include "utils.h"

__global__ void MatrixMulKernel(float *M, float *N, float *P, int rows,
                                int cols, int Width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((row < rows) && (col < cols)) {
    float Pvalue = 0;
    for (int k = 0; k < Width; ++k) {
      Pvalue += M[row * Width + k] * N[k * Width + col];
    }
    P[row * cols + col] = Pvalue;
  }
}

void print_matrix(float *mat, int rows, int cols) {
  printf("[");
  for (int i = 0; i < rows; ++i) {
    printf("[");
    for (int j = 0; j < cols; ++j) {
      printf("%f", mat[i * cols + j]);
      if (j != cols - 1) {
        printf(", ");
      }
    }

    if (i != rows - 1) {
      printf("]\n");
    } else {
      printf("]");
    }
  }
  printf("]\n");
}

int main() {
  int i = 16, j = 8, k = 32;
  // M_ij * N_jk = P_ik
  float *M = new float[i * j];
  float *N = new float[j * k];
  float *P = new float[i * k];

  for (int a = 0; a < i * j; ++a) {
    M[a] = (float)a;
  }

  for (int b = 0; b < j * k; ++b) {
    N[b] = (float)b;
  }

  float *d_M, *d_N, *d_P;
  HANDLE_ERROR(cudaMalloc((void **)&d_M, i * j * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_N, j * k * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_P, i * k * sizeof(float)));

  HANDLE_ERROR(
      cudaMemcpy(d_M, M, i * j * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(d_N, N, j * k * sizeof(float), cudaMemcpyHostToDevice));

  dim3 dimGrid(k, i);
  MatrixMulKernel<<<dimGrid, 1>>>(d_M, d_N, d_P, i, k, j);

  HANDLE_ERROR(
      cudaMemcpy(P, d_P, i * k * sizeof(float), cudaMemcpyDeviceToHost));

  printf("M=\n");
  print_matrix(M, i, j);
  printf("N=\n");
  print_matrix(N, j, k);
  printf("M * N = P\n");
  print_matrix(P, i, k);

  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);

  delete [] M;
  delete [] N;
  delete [] P;
  return 0;
}