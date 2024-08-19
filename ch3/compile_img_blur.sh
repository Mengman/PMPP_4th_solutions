#!/bin/bash

nvcc -o img_blur img_blur.cu \
    -I/usr/local/include/opencv4 -I../utils \
    -L/usr/local/lib -lopencv_core -lopencv_imgcodecs
