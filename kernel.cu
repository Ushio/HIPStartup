#include "typedbuffer.hpp"

__device__
int sqr(int x)
{
	return x * x;
}

extern "C" __global__ void kernelMain( TypedBuffer<int> xs )
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	xs[tid] = sqr(tid);
}