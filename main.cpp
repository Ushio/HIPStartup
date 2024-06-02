#include <iostream>
#include <memory>
#include <Orochi/Orochi.h>
#include "shader.hpp"

int main() {
	if (oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0))
	{
		printf("failed to init..\n");
		return 0;
	}
	int deviceIdx = 0;

	oroError err;
	err = oroInit(0);
	oroDevice device;
	err = oroDeviceGet(&device, deviceIdx);
	oroCtx ctx;
	err = oroCtxCreate(&ctx, 0, device);
	oroCtxSetCurrent(ctx);

	oroStream stream = 0;
	oroStreamCreate(&stream);
	oroDeviceProp props;
	oroGetDeviceProperties(&props, device);

	bool isNvidia = oroGetCurAPI(0) & ORO_API_CUDADRIVER;

	printf("Device: %s\n", props.name);
	printf("Cuda: %s\n", isNvidia ? "Yes" : "No");

	int blockDim = 32;
	int blocks = 4;
	void* inputs;
	oroMalloc(&inputs, sizeof(int) * blockDim * blocks );

	{
		std::string baseDir = "../"; /* repository root */

		std::vector<std::string> options;
		if (isNvidia)
		{
			options.push_back("--gpu-architecture=compute_70");
			options.push_back(NV_ARG_LINE_INFO);
		}
		else
		{
			options.push_back(AMD_ARG_LINE_INFO);
		}

		// Debug
		if( isNvidia )
		{
			options.push_back("-G");
		}
		else
		{
			options.push_back("-O0");
		}

		Shader shader((baseDir + "\\kernel.cu").c_str(), "kernel.cu", { baseDir }, options );

		int val = 4;
		shader.launch("kernelMain",
			ShaderArgument().value(inputs),
			blocks, 1, 1, blockDim, 1, 1, stream);

		oroStreamSynchronize(stream);
	}

	std::vector<int> outputs(blockDim * blocks);
	oroMemcpyDtoH(outputs.data(), inputs, sizeof(int) * blockDim * blocks);

	for (auto tid : outputs)
	{
		printf("%d\n", tid);
	}

	oroFree(inputs);

	return 0;
}
