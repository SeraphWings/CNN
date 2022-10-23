#include "layer.h"

// Constructor
Layer::Layer(int M, int N, int O)
{
	this->M = M;
	this->N = N;
	this->O = O;


}

// Destructor
Layer::~Layer()
{

}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
	
}

// Reset GPU memory between iterations
void Layer::clear()
{
	
}



/*
__device__ float step_function(float v)
{
	//sinsoid function
	return 1 / (1 + exp(-v));
}

__global__ void apply_step_function(float *input, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = step_function(input[idx]);
	}

	printf("output: ");
			for (int i = 0; i < 10; i++)
			{
				printf("%.3f ",output[i]);
				
			}
	printf("\n");

}

__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	// printf("output:");
	// 		for (int i = 0; i < 10; i++)
	// 		{
	// 			printf("%.3f ",output[i]);
				
	// 		}
	// printf("\n");
	
	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
		//printf("%.3f ",err[idx]);
	}
	//printf("\n");
	
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] += dt * grad[idx];
	}
}
*/