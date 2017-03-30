#include <vector>
#include <stdio.h>

#include "boost/thread/thread.hpp"

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class MyTest : public ::testing::Test{};

TEST_F(MyTest, TestAsyncRecycle_atGPU) {
  cudaStream_t stream0;
  cudaStreamCreate(&stream0);
  Caffe::set_mode(Caffe::GPU);
  SyncedMemory mem(100000000);
  SyncedMemory mem1(100000000);
  printf("%d\n", Caffe::mode() == Caffe::GPU);
  void * cpu_data = mem.mutable_cpu_data();
  caffe_memset(mem.size(), 1, cpu_data);
  cpu_data = mem1.mutable_cpu_data();
  caffe_memset(mem1.size(), 1, cpu_data);
  const void * gpu_data = mem.gpu_data();

  cudaDeviceSynchronize();
  mem.mutable_gpu_data();
#pragma omp parallel sections
  {
#pragma omp section
	  {
		  mem.recycle_gpu_data(stream0);
	  }
#pragma omp section
	  {
		  mem1.gpu_data();
	  }
  }
  cudaDeviceSynchronize();
}

TEST_F(MyTest, TestAsyncRecycle_Synced) {
  cudaStream_t stream0;
  cudaStreamCreate(&stream0);
  Caffe::set_mode(Caffe::GPU);
  SyncedMemory mem(100000000);
  SyncedMemory mem1(100000000);
  printf("%d\n", Caffe::mode() == Caffe::GPU);
  void * cpu_data = mem.mutable_cpu_data();
  caffe_memset(mem.size(), 1, cpu_data);
  cpu_data = mem1.mutable_cpu_data();
  caffe_memset(mem1.size(), 1, cpu_data);
  const void * gpu_data = mem.gpu_data();

  cudaDeviceSynchronize();
#pragma omp parallel sections
  {
#pragma omp section
	  {
		  mem.recycle_gpu_data(stream0);
	  }
#pragma omp section
	  {
		  mem1.gpu_data();
	  }
  }
  cudaDeviceSynchronize();
}

}
