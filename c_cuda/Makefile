##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: Makefile
#
# Description: Simple Makefile.  Run 'make clean && make all'
# 					This requires make HDF5, OpenMP, and Nvidia CUDA toolkit (and gcc and make)
#
##############################################################################

CC = gcc
CXX ?= g++
NVCC = nvcc
IDIR = .
BUILD_DIR ?= .
CUDA_ARCH ?= native

#set the HDF5 env variables for your system

#on Ubuntu (install libhdf5-dev with aptitude)
HDF5_INC = /usr/include/hdf5/serial/
HDF5_LIB = /usr/lib/$(shell $(CC) -dumpmachine)/hdf5/serial/

#on Arch (install hdf5 with pacman)
#HDF5_INC = /usr/include/
#HDF5_LIB = /usr/lib/

#on Centos (install hdf5-devel with yum)
#HDF5_INC = /usr/include/
#HDF5_LIB = /usr/lib64/

#if you download and extract shared libraries here, use something like this
#HDF5_INC ?= ./hdf5/include/
#HDF5_LIB ?= ./hdf5/lib/ #will also need to add this to LD_LIBRARY_PATH when running .x


CFLAGS = -I$(IDIR) -Wall -Wextra -Wundef -Wpedantic -std=c99
CFLAGS += -I$(HDF5_INC)
CFLAGS += -O3
LDFLAGS = -fopenmp -lgomp -lm
LDFLAGS += -L$(HDF5_LIB) -lhdf5

NVCCFLAGS = -I$(IDIR) -O3 -arch=$(CUDA_ARCH)
NVCCFLAGS += -Xptxas -O3 -lineinfo
NVCCFLAGS += -x cu #compile .c or .h as cuda
NVCCFLAGS += -I$(HDF5_INC)
#NVCCFLAGS += -ptx
NVCCFLAGS += -Xcompiler -Wall -Xcompiler -Wextra #not using -Wpedantic -Wundef with nvcc
CULDFLAGS = -Xcompiler -fopenmp -lgomp -lm
CULDFLAGS += -L$(HDF5_LIB) -lhdf5

SRC= fdtd_main.c

cpu: cpu_double cpu_single 

gpu: gpu_double gpu_single 

all: cpu gpu

#build cpu/gpu double/single precision binaries
cpu_double:
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRC) -o $(BUILD_DIR)/fdtd_main_cpu_double.x -DPRECISION=2 -DUSING_CUDA=false $(LDFLAGS)

cpu_single:
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRC) -o $(BUILD_DIR)/fdtd_main_cpu_single.x -DPRECISION=1 -DUSING_CUDA=false $(LDFLAGS)

gpu_double:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(BUILD_DIR)/fdtd_main_gpu_double.x -DPRECISION=2 -DUSING_CUDA=true $(CULDFLAGS)

gpu_single:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(BUILD_DIR)/fdtd_main_gpu_single.x -DPRECISION=1 -DUSING_CUDA=true $(CULDFLAGS)

test-halo:
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Wpedantic -I$(IDIR) ../tests/native/halo_faces_test.cpp -o $(BUILD_DIR)/halo_faces_test.x
	$(BUILD_DIR)/halo_faces_test.x

test-boundary:
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) ../tests/boundary_fd_check.c -o $(BUILD_DIR)/boundary_fd_single_test.x -DPRECISION=1 -DUSING_CUDA=false $(LDFLAGS)
	OMP_NUM_THREADS=2 $(BUILD_DIR)/boundary_fd_single_test.x
	$(CC) $(CFLAGS) ../tests/boundary_fd_check.c -o $(BUILD_DIR)/boundary_fd_double_test.x -DPRECISION=2 -DUSING_CUDA=false $(LDFLAGS)
	OMP_NUM_THREADS=2 $(BUILD_DIR)/boundary_fd_double_test.x

test-native: test-halo test-boundary

build-cuda-tests:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -std=c++11 ../tests/native/halo_faces_test.cpp -o $(BUILD_DIR)/halo_faces_cuda_test.x
	$(NVCC) $(NVCCFLAGS) -std=c++11 ../tests/boundary_fd_cuda_check.cu -o $(BUILD_DIR)/boundary_fd_single_cuda_test.x -DPRECISION=1 -DUSING_CUDA=true $(CULDFLAGS)
	$(NVCC) $(NVCCFLAGS) -std=c++11 ../tests/boundary_fd_cuda_check.cu -o $(BUILD_DIR)/boundary_fd_double_cuda_test.x -DPRECISION=2 -DUSING_CUDA=true $(CULDFLAGS)

# Tests return 77 when no CUDA device is visible; preserve that result.
test-cuda: build-cuda-tests
	$(BUILD_DIR)/halo_faces_cuda_test.x
	$(BUILD_DIR)/boundary_fd_single_cuda_test.x
	$(BUILD_DIR)/boundary_fd_double_cuda_test.x

.PHONY: all cpu gpu cpu_double cpu_single gpu_double gpu_single test-halo test-boundary test-native build-cuda-tests test-cuda clean

clean:
	rm -f $(BUILD_DIR)/*.x
