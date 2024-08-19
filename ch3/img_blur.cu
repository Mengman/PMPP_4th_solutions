#include <opencv2/opencv.hpp>
#include "utils.h"

// #define BLUR_SIZE 1
// set to 3 to make the blur effect more visible
#define BLUR_SIZE 3

using namespace cv;

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w,
                           int h, int ch) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < w && row < h) {
    int pixB = 0, pixG = 0, pixR = 0;
    int pixels = 0;

    for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
      for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
        int curRow = row + blurRow;
        int curCol = col + blurCol;
        if (curCol >= 0 && curCol < w && curRow >= 0 && curRow < h) {
          int curOffset = (curRow * w + curCol) * ch;
          pixB += in[curOffset];
          pixG += in[curOffset + 1];
          pixR += in[curOffset + 2];
          ++pixels;
        }
      }
    }
    int offset = (row * w + col) * 3;
    out[offset] = (unsigned char)(pixB / pixels);
    out[offset + 1] = (unsigned char)(pixG / pixels);
    out[offset + 2] = (unsigned char)(pixR / pixels);
  }
}

int main() {
  printf("read image from lenna.png\n");
  Mat img = imread("lenna.png", IMREAD_COLOR);
  int rows = img.rows;
  int cols = img.cols;
  int channels = img.channels();
  unsigned char *blur_data = new unsigned char[rows * cols * channels];

  unsigned char *d_img_data, *d_blur_data;
  HANDLE_ERROR(cudaMalloc((void **)&d_img_data,
                          rows * cols * channels * sizeof(unsigned char)));
  HANDLE_ERROR(cudaMalloc((void **)&d_blur_data,
                          rows * cols * channels * sizeof(unsigned char)));

  HANDLE_ERROR(cudaMemcpy(d_img_data, img.data,
                          rows * cols * channels * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

  dim3 dimBlock(32, 32);
  dim3 dimGrid(ceil(cols / 32.0), ceil(rows / 32.0));
  blurKernel<<<dimGrid, dimBlock>>>(d_img_data, d_blur_data, cols, rows, channels);

  HANDLE_ERROR(cudaMemcpy(blur_data, d_blur_data,
                          rows * cols * channels * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));

  Mat blurImg(rows, cols, CV_8UC3, blur_data);

  printf("save blur image to 'lenna_blur.jpg'\n");
  imwrite("lenna_blur.jpg", blurImg);

  cudaFree(d_img_data);
  cudaFree(d_blur_data);

  delete[] blur_data;

  return 0;
}