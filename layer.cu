#include "layer.h"

// Constructor
Layer::Layer(int M, int N, int O)
{
	this->C = M;
	this->H = N;
	this->W = O;
	
	if(this->C != 1 && this->H  != 1 && this->W != 1 ){
		//3D layer
		this-> data3D = new double** [this->C];
		for (int i = 0; i < this->C; i++)
		{
			data3D[i] = new double* [this->H];
			for (int j = 0; j < this->H; j++)
			{
				data3D[i][j] = new double [this->W];
			}
			
		}
	}
	else if(this->C == 1 && this->H  != 1 && this->W != 1 )
	{
		//2D
		this-> data2D = new double* [this->H];
		for (int i = 0; i < this->H; i++)
		{
			data2D[i] = new double[this->W];
		}
		
	}
	else if (this->C == 1 && this->H  == 1 && this->W != 1 )
	{
		//1D
		this-> data1D = new double[this->W];
	}
	
}

// Destructor
Layer::~Layer()
{
	free(this->data1D);
	free(this->data2D);
	free(this->data3D);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	this->data1D = nullptr;
	this->data2D = nullptr;
	this->data3D = nullptr;

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

void Layer::conv2D(double** input){
	//printf("conv start\n");
	//2*2 6 kernel 0 1 0 1
	
	for (int i = 0; i < 27; i++)
	{
		for (int j = 0; j < 27; j++)
		{
			this->data3D[0][i][j] = input[i][j] * 0.0 + input[i][j+1] * 1.0 + input[i+1][j] * 0.0 + input[i+1][j+1] * 1.0;
			this->data3D[1][i][j] = input[i][j] * 1.0 + input[i][j+1] * 0.0 + input[i+1][j] * 1.0 + input[i+1][j+1] * 0.0;
			this->data3D[2][i][j] = input[i][j] * 0.0 + input[i][j+1] * 0.0 + input[i+1][j] * 1.0 + input[i+1][j+1] * 1.0;
			this->data3D[3][i][j] = input[i][j] * 1.0 + input[i][j+1] * 1.0 + input[i+1][j] * 0.0 + input[i+1][j+1] * 0.0;
			this->data3D[4][i][j] = input[i][j] * 0.0 + input[i][j+1] * 1.0 + input[i+1][j] * 1.0 + input[i+1][j+1] * 1.0;
			this->data3D[5][i][j] = input[i][j] * 1.0 + input[i][j+1] * 1.0 + input[i+1][j] * 1.0 + input[i+1][j+1] * 0.0;
			//this->data2D[i][j] = 0.0;
			//printf("%.2lf ",this->data2D[i][j]);
		}
		//printf("\n");
		
	}

	//printf("conv end\n");
}

void Layer::maxPooling(double **input){
	//2*2 max
	
	for (int i = 0; i < 26; i++)
	{
		for (int j = 0; j < 26; j++)
		{
			double max = input[i][j];
			for (int k = 0; k < 2; k++)
			{
				for (int l = 0; l < 2; l++)
				{
					if(max <= input[i+k][j+l]){
						max = input[i+k][j+l];
					}
				}
				
				
			}
			
			this->data2D[i][j] = max;
		}
		
	}
	
	
}

void Layer::maxPooling(double ***input){
	//2*2 max
	
	for (int i = 0; i < 26; i++)
	{
		for (int j = 0; j < 26; j++)
		{
			double max0 = input[0][i][j];
			double max1 = input[1][i][j];
			double max2 = input[2][i][j];
			double max3 = input[3][i][j];
			double max4 = input[4][i][j];
			double max5 = input[5][i][j];
			for (int k = 0; k < 2; k++)
			{
				for (int l = 0; l < 2; l++)
				{
					if(max0 <= input[0][i+k][j+l]){
						max0 = input[0][i+k][j+l];
					}
					if(max1 <= input[1][i+k][j+l]){
						max1 = input[1][i+k][j+l];
					}
					if(max2 <= input[2][i+k][j+l]){
						max2 = input[2][i+k][j+l];
					}
					if(max3 <= input[3][i+k][j+l]){
						max3 = input[3][i+k][j+l];
					}
					if(max4 <= input[4][i+k][j+l]){
						max4 = input[4][i+k][j+l];
					}
					if(max5 <= input[5][i+k][j+l]){
						max5 = input[5][i+k][j+l];
					}
				}
				
				
			}
			
			this->data3D[0][i][j] = max0;
			this->data3D[1][i][j] = max1;
			this->data3D[2][i][j] = max2;
			this->data3D[3][i][j] = max3;
			this->data3D[4][i][j] = max4;
			this->data3D[5][i][j] = max5;
		}
		
	}
	
	
}

void Layer::flatten(double **input){
	
	printf("2D flatten\n");
	
	for (int i = 0; i < this->H; i++)
	{
		for (int j = 0; j < this->W; j++)
		{
			this->data1D[i * this->W + j] = input[i][j];
		}
		
	}
	

	//memcpy(this->data1D, input, this->H * this->W * sizeof(double));



	
}


void Layer::flatten(double ***input){

	//printf("3D flatten\n");

	for (int i = 0; i < this->C; i++)
	{
		for (int j = 0; j < this->H; j++)
		{
			for (int k = 0; k < this->W; k++)
			{
				this->data1D[i * this->H * this->W + j * this->W + k] = input[i][j][k];
			}
			
			
		}
		
	}
	
}


void Layer::in_hidden(double *input, double kernel[26*26][343], double bias[343] ){

	// printf("in hidden\n");
	// for (int i = 0; i < 676; i++)
	// {
	// 	printf("%.2lf ", input[i]);
	// }
	// printf("\n");

	// printf("in hidden kernel\n");
	// for (int i = 0; i < 676; i++)
	// {
	// 	printf("%.2lf ", kernel[i][0]);
	// }
	// printf("\n");

	// printf("initiated\n");
	// for (int i = 0; i < 343; i++)
	// {
	// 	printf("%.2lf ", this->data1D[i]);
	// }
	// printf("\n");

	//dot
	for (int i = 0; i < 343; i++)
	{
		this->data1D[i] = 0.0;
		for (int j = 0; j < 26*26; j++)
		{
			this->data1D[i] += input[j] * kernel[j][i];
		}
		
	}

	//bias
	
	for (int i = 0; i < 343; i++)
	{
		this->data1D[i] += bias[i];
	}
	
	
	//RELU 
	for (int i = 0; i < 343; i++)
	{
		this->data1D[i] = this->data1D[i] > 0.0?this->data1D[i]:0.0; 
	}

	

	/*
	//softmax as activate function
	double sum = 0.0;
	double max = this->data1D[0];
	for (int i = 0; i < 343; i++)
	{
		if(max < this->data1D[i]){
			max = this->data1D[i];
		}

	}

	for (int i = 0; i < 343; i++)
	{
		
		sum += exp(this->data1D[i] - max) ;

	}

	for (int i = 0; i < 343; i++)
	{
		this->data1D[i] = ceil( ( exp(this->data1D[i] - max)/sum ) * 100) / 100;
	}
	*/

}

void Layer::in_hidden(double *input, double kernel[26*26*6][2033], double bias[2033] ){

	// printf("in hidden\n");
	// for (int i = 0; i < 676; i++)
	// {
	// 	printf("%.2lf ", input[i]);
	// }
	// printf("\n");

	// printf("in hidden kernel\n");
	// for (int i = 0; i < 676; i++)
	// {
	// 	printf("%.2lf ", kernel[i][0]);
	// }
	// printf("\n");

	// printf("initiated\n");
	// for (int i = 0; i < 343; i++)
	// {
	// 	printf("%.2lf ", this->data1D[i]);
	// }
	// printf("\n");

	//dot
	for (int i = 0; i < 2033; i++)
	{
		this->data1D[i] = 0.0;
		for (int j = 0; j < 26*26*6; j++)
		{
			this->data1D[i] += input[j] * kernel[j][i];
		}
		
	}

	//bias
	
	for (int i = 0; i < 2033; i++)
	{
		this->data1D[i] += bias[i];
	}
	
	
	//RELU 
	for (int i = 0; i < 2033; i++)
	{
		this->data1D[i] = this->data1D[i] > 0.0?this->data1D[i]:0.0; 
	}

	

	/*
	//softmax as activate function
	double sum = 0.0;
	double max = this->data1D[0];
	for (int i = 0; i < 343; i++)
	{
		if(max < this->data1D[i]){
			max = this->data1D[i];
		}

	}

	for (int i = 0; i < 343; i++)
	{
		
		sum += exp(this->data1D[i] - max) ;

	}

	for (int i = 0; i < 343; i++)
	{
		this->data1D[i] = ceil( ( exp(this->data1D[i] - max)/sum ) * 100) / 100;
	}
	*/

}

void Layer::dense(double *input, double kernel[2033][10], double bias[10]){
	
	for (int i = 0; i < 10; i++)
	{
		this->data1D[i] = 0.0;
	}

	//dot
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 2033; j++)
		{
			this->data1D[i] += input[j] * kernel[j][i];
		}
		
	}

	//bias
	
	for (int i = 0; i < 10; i++)
	{
		this->data1D[i] += bias[i];
	}
	
	/*
	//RELU
	for (int i = 0; i < 10; i++)
	{
		output.data1D[i] = output.data1D[i] > 0.0?output.data1D[i]:0.0; 
	}

	*/

	//softmax as activate function
	double sum = 0.0;
	double max = this->data1D[0];
	for (int i = 0; i < 10; i++)
	{
		if(max < this->data1D[i]){
			max = this->data1D[i];
		}

	}

	for (int i = 0; i < 10; i++)
	{
		
		sum += exp(this->data1D[i] - max) ;

	}

	for (int i = 0; i < 10; i++)
	{
		this->data1D[i] = ceil( ( exp(this->data1D[i] - max)/sum ) * 100) / 100;
	}

}

void Layer::printData(){
	if (this->data1D != nullptr)
	{
		printf("%d 1D found \n", this->W);
		for (int i = 0; i < this->W; i++)
		{
			printf("%.2lf ", this->data1D[i]);
		}
		printf("\n");
	}
	else if (this->data2D != nullptr)
	{
		printf("%d %d 2D found \n", this->H, this->W);
		for (int i = 0; i < this->H; i++)
		{
			for (int j = 0; j < this->W; j++)
			{
				printf("%.2lf ", this->data2D[i][j]);
			}
			printf("\n");
			
		}
		printf("\n");
	}
	else if (this->data3D != nullptr)
	{
		printf("3D found \n");
		for (int i = 0; i < this->H; i++)
		{
			for (int j = 0; j < this->W; j++)
			{
				printf("%.2lf ", this->data3D[0][i][j]);
			}
			printf("\n");
			
		}
		printf("\n");
	}
	else{
		printf("all data are nullptr \n");
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