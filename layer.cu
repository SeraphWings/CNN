#include "layer.h"

// Constructor
Layer::Layer(int M, int N, int O)
{
	this->C = M;
	this->H = N;
	this->W = O;

	if(M != 1){
		//3D layer
		data3D = new double** [M];
		for (int i = 0; i < M; i++)
		{
			data3D[i] = new double* [N];
			for (int j = 0; j < N; j++)
			{
				data3D[i][j] = new double [O];
			}
			
		}
	}
	else if (M == 1 && N != 1)
	{
		//2D
		data2D = new double* [N];
		for (int i = 0; i < N; i++)
		{
			data2D[i] = new double[O];
		}
		
	}
	else if (M == 1 && N == 1)
	{
		//1D
		data1D = new double[O];
	}
	
	
	

}

// Destructor
Layer::~Layer()
{

}

// Reset GPU memory between iterations
void Layer::clear()
{
	
}

// Send data one row from dataset to the GPU
/*
void Layer::setOutput(float *data)
{
	
}
*/

void Layer::readInput(double input[28][28]){

	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			this->data2D[i][j] = input[i][j];
		}
		
	}
	
}

Layer Layer::conv2D(){
	//2*2 kernel 0 1 0 1
	Layer output(1,27,27);

	for (int i = 0; i < 27; i++)
	{
		for (int j = 0; j < 27; j++)
		{
			output.data2D[i][j] = this->data2D[i][j] * 0 + this->data2D[i][j+1] * 1 + this->data2D[i+1][j] * 0 + this->data2D[i+1][j+1] * 1;
		}
		
	}

	return output;
}

Layer Layer::maxPooling(){
	//2*2 max
	Layer output(1,26,26);

	for (int i = 0; i < 26; i++)
	{
		for (int j = 0; j < 26; j++)
		{
			double max = this->data2D[i][j];
			for (int k = 0; k < 2; k++)
			{
				for (int l = 0; l < 2; l++)
				{
					if(max < this->data2D[i+k][j+l]){
						max = this->data2D[i+k][j+l];
					}
				}
				
				
			}
			
			output.data2D[i][j] = max;
		}
		
	}

	return output;
}

Layer Layer::flatten(){
	
	if(this-> data2D != nullptr){
		Layer output(1,1, this->H * this->W);

		for (int i = 0; i < this->H; i++)
		{
			for (int j = 0; j < this->W; j++)
			{
				output.data1D[i * this->H + j] = this->data2D[i][j];
			}
			
		}
		return output;
	}
	else if (this-> data3D != nullptr)
	{
		Layer output(1,1, this->C * this->H * this->W);

		for (int i = 0; i < this->C; i++)
		{
			for (int j = 0; j < this->H; j++)
			{
				for (int k = 0; k < this->W; k++)
				{
					output.data1D[i * this->H * this-> W + j * this-> H + k] = this->data2D[i][j];
				}
				
				
			}
			
		}
		return output;
	}

	
	
}

Layer Layer::Dense(){
	Layer output(1,1,10);

	

	return output;
}

void Layer::printData(){
	if (this->data1D != nullptr)
	{
		for (int i = 0; i < this->W; i++)
		{
			printf("%.2lf ", data1D[i]);
		}
		printf("\n");
	}
	else if (this->data2D != nullptr)
	{
		for (int i = 0; i < this->H; i++)
		{
			for (int j = 0; j < this->W; j++)
			{
				printf("%.2lf ", data2D[i][j]);
			}
			printf("\n");
			
		}
		printf("\n");
	}
	
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