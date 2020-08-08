#ifndef GPU_GRAPH_ALGORITHMS_KERNELS_CUH
#define GPU_GRAPH_ALGORITHMS_KERNELS_CUH
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <stdio.h>
#include <string>
#include <string.h>
#include <ctime>

using std::cout;
using std::endl;

__global__ void relax(int N, int MAX_VAL, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_Di, int *d_out_P,  int *d_out_Pi);
__global__ void updateDistance(int N, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_Di, int *d_out_P,  int *d_out_Pi);
__global__ void updateIndexOfEdges(int N, int *d_in_V, int *d_in_E, int l, int r);
__global__ void initializeArray(const int N, int *p, const int val, bool sourceDifferent, const int source, const int sourceVal);

__global__ void relaxWithGridStride(int N, int MAX_VAL, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_Di, int *d_out_P,  int *d_out_Pi);
__global__ void updateDistanceWithGridStride(int N, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_Di, int *d_out_P,  int *d_out_Pi);
__global__ void updateIndexOfEdgesWithGridStide(int N, int *d_in_V, int *d_in_E, int l, int r);
__global__ void initializeArrayWithGridStride(const int N, int *p, const int val, bool sourceDifferent, const int source, const int sourceVal);
#endif //GPU_GRAPH_ALGORITHMS_KERNELS_CUH
