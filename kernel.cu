#include "typedbuffer.hpp"
#include "helper_math.h"

__device__
int sqr(int x)
{
	return x * x;
}

extern "C" __global__ void kernelMain( TypedBuffer<int> xs, int value )
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	xs[tid] = sqr(tid) + value;
}