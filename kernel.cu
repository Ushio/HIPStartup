
extern "C" __global__ void kernelMain( int* xs )
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	xs[tid] = tid;
}