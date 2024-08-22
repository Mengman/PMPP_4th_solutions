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

__global__ void MatMulRowKernel(float *M, float *N, float *P, int rows,
                                int cols, int Width) {
  int row = threadIdx.x;
  if (row < rows) {
    for (int col = 0; col < cols; ++col) {
      int Pvalue = 0;
      for (int k = 0; k < Width; ++k) {
        Pvalue += M[row * Width + k] * N[k * Width + col];
      }
      P[row * cols + col] = Pvalue;
    }
  }
}

__global__ void MatMulColKernel(float *M, float *N, float *P, int rows,
                                int cols, int Width) {
  int col = threadIdx.x;
  if (col < cols) {
    for (int row = 0; row < rows; ++row) {
      int Pvalue = 0;
      for (int k = 0; k < Width; ++k) {
        Pvalue += M[row * Width + k] * N[k * Width + col];
      }
      P[row * cols + col] = Pvalue;
    }
  }
}

int main() {
//   int i = 16, j = 8, k = 32;
  int i = 32, j = 32, k = 32;
  // M_ij * N_jk = P_ik
  float *M = new float[i * j];
  float *N = new float[j * k];
  float *P = new float[i * k];
  float *Prow = new float[i * k];
  float *Pcol = new float[i * k];

  for (int a = 0; a < i * j; ++a) {
    M[a] = (float)a;
  }

  for (int b = 0; b < j * k; ++b) {
    N[b] = (float)b;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  float *d_M, *d_N, *d_P, *d_Prow, *d_Pcol;
  HANDLE_ERROR(cudaMalloc((void **)&d_M, i * j * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_N, j * k * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_P, i * k * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_Prow, i * k * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_Pcol, i * k * sizeof(float)));

  HANDLE_ERROR(
      cudaMemcpy(d_M, M, i * j * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(d_N, N, j * k * sizeof(float), cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  dim3 dimGrid(k, i);
  MatrixMulKernel<<<dimGrid, 1>>>(d_M, d_N, d_P, i, k, j);
  cudaEventRecord(stop);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  printf("MatrixMulKernel time elapsed: %f\n", elapsedTime);
  elapsedTime = 0.0;

  HANDLE_ERROR(
      cudaMemcpy(P, d_P, i * k * sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  MatMulRowKernel<<<1, i>>>(d_M, d_N, d_Prow, i, k, j);
  cudaEventRecord(stop);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("MatMulRowKernel time elapsed: %f\n", elapsedTime);
  elapsedTime = 0.0;

  HANDLE_ERROR(
      cudaMemcpy(Prow, d_Prow, i * k * sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  MatMulColKernel<<<1, k>>>(d_M, d_N, d_Pcol, i, k, j);
  cudaEventRecord(stop);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("MatMulColKernel time elapsed: %f\n", elapsedTime);

  HANDLE_ERROR(
      cudaMemcpy(Pcol, d_Pcol, i * k * sizeof(float), cudaMemcpyDeviceToHost));

  for (int a = 0; a < i; ++a) {
    for (int b = 0; b < k; ++b) {
      int idx = a * k + b;
      if (P[idx] != Prow[idx]) {
        printf("Prow[%d,%d]=%f not equal to P[%d,%d]=%f\n", a, b, Prow[idx], a,
               b, P[idx]);
      }

      if (P[idx] != Pcol[idx]) {
        printf("Pcol[%d,%d]=%f not equal to P[%d,%d]=%f\n", a, b, Pcol[idx], a,
               b, P[idx]);
      }
    }
  }

  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);
  cudaFree(d_Prow);
  cudaFree(d_Pcol);

  delete[] M;
  delete[] N;
  delete[] P;
  delete[] Pcol;
  delete[] Prow;
  return 0;
}