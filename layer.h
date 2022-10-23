#include <cstdlib>
#include <vector>
#include <memory>
#include <cmath>
#include <cuda.h>


#ifndef LAYER_H
#define LAYER_H
#endif


//float delta = 1.0E-01f;
const static float epsilon = 1.0E-02f;
// const static float epsilon = 0.1f;

class Layer {
	public:
	int M, N, O;



	Layer(int M, int N, int O);

	~Layer();

	void setOutput(float *data);
	void clear();
};


// Utility CUDA kernel functions
// __device__ float step_function(float v);
// __global__ void apply_step_function(float *input, float *output, const int N);
// __global__ void makeError(float *err, float *output, unsigned int Y, const int N);
// __global__ void apply_grad(float *output, float *grad, const int N);


// Forward propagation kernels


// Back propagation kernels

