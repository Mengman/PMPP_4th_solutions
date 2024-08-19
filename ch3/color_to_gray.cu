#include <opencv2/opencv.hpp>
#include <utils.h>

using namespace cv;

__global__ void colortoGrayscaleConvertion(unsigned char *Pout,
                                           unsigned char *Pin, int width,
                                           int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * 3;
    // opencv pixel order BGR
    unsigned char b = Pin[rgbOffset];
    unsigned char g = Pin[rgbOffset + 1];
    unsigned char r = Pin[rgbOffset + 2];

    Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}

int main() {
  printf("read image from 'lenna_small.jpg'\n");
  Mat img = imread("lenna_small.jpg", IMREAD_COLOR);
  int rows = img.rows;
  int cols = img.cols;
  int channels = img.channels();
  unsigned char *gray_data = new unsigned char[rows * cols];

  unsigned char *d_img_data, *d_gray_data;

  // allocate cuda global memory for d_image_data and d_gray_data
  HANDLE_ERROR(cudaMalloc((void **)&d_img_data,
                          rows * cols * channels * sizeof(unsigned char)));
  HANDLE_ERROR(
      cudaMalloc((void **)&d_gray_data, rows * cols * sizeof(unsigned char)));

  // copy image to GPU
  HANDLE_ERROR(cudaMemcpy(d_img_data, img.data,
                          rows * cols * channels * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

  dim3 dimGrid(5, 4);
  dim3 dimBlock(16, 16);
  colortoGrayscaleConvertion<<<dimGrid, dimBlock>>>(d_gray_data, d_img_data,
                                                    cols, rows);

  HANDLE_ERROR(cudaMemcpy(gray_data, d_gray_data,
                          rows * cols * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));

  Mat grayImage(rows, cols, CV_8UC1, gray_data);

  printf("save image to 'lenna_small_gray.jpg'\n");
  imwrite("lenna_small_gray.jpg", grayImage);

  cudaFree(d_img_data);
  cudaFree(d_gray_data);

  delete[] gray_data;

  return 0;
}