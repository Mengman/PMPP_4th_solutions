#!/bin/bash

nvcc -o color_to_gray color_to_gray.cu \
    -I/usr/local/include/opencv4 -I../utils \
    -L/usr/local/lib -lopencv_core -lopencv_imgcodecs
