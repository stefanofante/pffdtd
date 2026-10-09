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
BUILD_REVISION ?= $(shell git rev-parse --short=12 HEAD 2>/dev/null)$(if $(shell git status --porcelain --untracked-files=no 2>/dev/null),-dirty,)

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

benchmark: benchmark_double benchmark_single

post: post_cpu post_gpu

post_cpu:
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++11 -O3 -Wall -Wextra -Wpedantic -ffp-contract=off -I$(IDIR) -I$(HDF5_INC) fdtd_post.cpp -o $(BUILD_DIR)/fdtd_post_cpu.x -DPFFDTD_BUILD_REVISION='"$(BUILD_REVISION)"' $(LDFLAGS)

post_gpu:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -std=c++11 --fmad=false fdtd_post.cpp -o $(BUILD_DIR)/fdtd_post_gpu.x -DPFFDTD_BUILD_REVISION='"$(BUILD_REVISION)"' $(CULDFLAGS)

benchmark_double:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -std=c++11 fdtd_bench.cu -o $(BUILD_DIR)/fdtd_bench_gpu_double.x -DPRECISION=2 -DUSING_CUDA=true -DPFFDTD_BUILD_REVISION='"$(BUILD_REVISION)"' $(CULDFLAGS)

benchmark_single:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -std=c++11 fdtd_bench.cu -o $(BUILD_DIR)/fdtd_bench_gpu_single.x -DPRECISION=1 -DUSING_CUDA=true -DPFFDTD_BUILD_REVISION='"$(BUILD_REVISION)"' $(CULDFLAGS)

fixture:
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off -I$(IDIR) -I$(HDF5_INC) fdtd_fixture.cpp -o $(BUILD_DIR)/fdtd_fixture.x -DPRECISION=2 -DUSING_CUDA=false $(LDFLAGS)

test-fixture: fixture
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off -I$(IDIR) -I$(HDF5_INC) fdtd_fixture.cpp -o $(BUILD_DIR)/fdtd_fixture_single.x -DPRECISION=1 -DUSING_CUDA=false $(LDFLAGS)
	pffdtd_fixture_parent=$$(mktemp -d "$(BUILD_DIR)/fixture-check.XXXXXX") && \
	$(BUILD_DIR)/fdtd_fixture_single.x --self-test "$$pffdtd_fixture_parent/fp1" && \
	$(BUILD_DIR)/fdtd_fixture.x --self-test "$$pffdtd_fixture_parent/fp2"

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

test-scheduler:
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++14 -O2 -Wall -Wextra -Wpedantic -I$(IDIR) ../tests/native/scheduler_plan_test.cpp -o $(BUILD_DIR)/scheduler_plan_test.x
	$(BUILD_DIR)/scheduler_plan_test.x

test-graph-plan:
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++14 -O2 -Wall -Wextra -Wpedantic -I$(IDIR) ../tests/native/graph_plan_test.cpp -o $(BUILD_DIR)/graph_plan_test.x
	$(BUILD_DIR)/graph_plan_test.x

test-boundary-map:
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++14 -O2 -Wall -Wextra -Wpedantic -I$(IDIR) ../tests/native/boundary_map_test.cpp -o $(BUILD_DIR)/boundary_map_test.x
	$(BUILD_DIR)/boundary_map_test.x

test-boundary-stencil:
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Wpedantic -frounding-math -ffp-contract=off -I$(IDIR) -I$(HDF5_INC) ../tests/native/boundary_stencil_test.cpp -o $(BUILD_DIR)/boundary_stencil_single_test.x -DPRECISION=1 -DUSING_CUDA=false $(LDFLAGS)
	OMP_NUM_THREADS=2 $(BUILD_DIR)/boundary_stencil_single_test.x
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Wpedantic -frounding-math -ffp-contract=off -I$(IDIR) -I$(HDF5_INC) ../tests/native/boundary_stencil_test.cpp -o $(BUILD_DIR)/boundary_stencil_double_test.x -DPRECISION=2 -DUSING_CUDA=false $(LDFLAGS)
	OMP_NUM_THREADS=2 $(BUILD_DIR)/boundary_stencil_double_test.x

test-ade-variants:
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off -I$(IDIR) -I$(HDF5_INC) ../tests/native/boundary_fd_specialized_test.cpp -o $(BUILD_DIR)/boundary_fd_specialized_single_test.x -DPRECISION=1 -DUSING_CUDA=false $(LDFLAGS)
	OMP_NUM_THREADS=2 $(BUILD_DIR)/boundary_fd_specialized_single_test.x
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off -I$(IDIR) -I$(HDF5_INC) ../tests/native/boundary_fd_specialized_test.cpp -o $(BUILD_DIR)/boundary_fd_specialized_double_test.x -DPRECISION=2 -DUSING_CUDA=false $(LDFLAGS)
	OMP_NUM_THREADS=2 $(BUILD_DIR)/boundary_fd_specialized_double_test.x

test-memory-launch:
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Wpedantic -I$(IDIR) ../tests/native/memory_launch_test.cpp -o $(BUILD_DIR)/memory_launch_test.x
	$(BUILD_DIR)/memory_launch_test.x

test-post:
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off -I$(IDIR) ../tests/native/post_dsp_test.cpp -o $(BUILD_DIR)/post_dsp_test.x
	$(BUILD_DIR)/post_dsp_test.x
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off -I$(IDIR) ../tests/native/post_resample_test.cpp -o $(BUILD_DIR)/post_resample_test.x
	$(BUILD_DIR)/post_resample_test.x
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off -I$(IDIR) ../tests/native/post_pipeline_test.cpp -o $(BUILD_DIR)/post_pipeline_test.x
	$(BUILD_DIR)/post_pipeline_test.x

test-native: test-halo test-boundary test-scheduler test-graph-plan test-boundary-map test-boundary-stencil test-ade-variants test-fixture test-memory-launch test-post

build-cuda-tests:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -std=c++11 --fmad=false ../tests/post_cuda_check.cu -o $(BUILD_DIR)/post_cuda_test.x $(CULDFLAGS)
	$(NVCC) $(NVCCFLAGS) -std=c++11 ../tests/native/halo_faces_test.cpp -o $(BUILD_DIR)/halo_faces_cuda_test.x
	$(NVCC) $(NVCCFLAGS) -std=c++11 ../tests/boundary_fd_cuda_check.cu -o $(BUILD_DIR)/boundary_fd_single_cuda_test.x -DPRECISION=1 -DUSING_CUDA=true $(CULDFLAGS)
	$(NVCC) $(NVCCFLAGS) -std=c++11 ../tests/boundary_fd_cuda_check.cu -o $(BUILD_DIR)/boundary_fd_double_cuda_test.x -DPRECISION=2 -DUSING_CUDA=true $(CULDFLAGS)
	$(NVCC) $(NVCCFLAGS) -std=c++11 ../tests/scheduler_cuda_check.cu -o $(BUILD_DIR)/scheduler_single_cuda_test.x -DPRECISION=1 -DUSING_CUDA=true $(CULDFLAGS)
	$(NVCC) $(NVCCFLAGS) -std=c++11 ../tests/scheduler_cuda_check.cu -o $(BUILD_DIR)/scheduler_double_cuda_test.x -DPRECISION=2 -DUSING_CUDA=true $(CULDFLAGS)
	$(NVCC) $(NVCCFLAGS) -std=c++11 ../tests/boundary_stencil_cuda_check.cu -o $(BUILD_DIR)/boundary_stencil_single_cuda_test.x -DPRECISION=1 -DUSING_CUDA=true $(CULDFLAGS)
	$(NVCC) $(NVCCFLAGS) -std=c++11 ../tests/boundary_stencil_cuda_check.cu -o $(BUILD_DIR)/boundary_stencil_double_cuda_test.x -DPRECISION=2 -DUSING_CUDA=true $(CULDFLAGS)

# Tests return 77 when no CUDA device is visible; preserve that result.
test-cuda: build-cuda-tests
	$(BUILD_DIR)/post_cuda_test.x
	$(BUILD_DIR)/halo_faces_cuda_test.x
	$(BUILD_DIR)/boundary_fd_single_cuda_test.x
	$(BUILD_DIR)/boundary_fd_double_cuda_test.x
	$(BUILD_DIR)/scheduler_single_cuda_test.x
	$(BUILD_DIR)/scheduler_double_cuda_test.x
	$(BUILD_DIR)/boundary_stencil_single_cuda_test.x
	$(BUILD_DIR)/boundary_stencil_double_cuda_test.x

.PHONY: all cpu gpu cpu_double cpu_single gpu_double gpu_single benchmark benchmark_double benchmark_single post post_cpu post_gpu fixture test-fixture test-halo test-boundary test-scheduler test-graph-plan test-boundary-map test-boundary-stencil test-ade-variants test-native test-memory-launch test-post build-cuda-tests test-cuda clean

clean:
	rm -f $(BUILD_DIR)/*.x
