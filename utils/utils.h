#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) __checkCudaError(err, __FILE__, __LINE__)

static void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))