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
	//2*2 1 kernel 0 1 0 1
	
	for (int i = 0; i < 27; i++)
	{
		for (int j = 0; j < 27; j++)
		{
			this->data2D[i][j] = input[i][j] * 0.0 + input[i][j+1] * 1.0 + input[i+1][j] * 0.0 + input[i+1][j+1] * 1.0;
			// [[[ 0.8025673 ]],[[-0.5093518 ]]],[[[-0.22089386]],[[-0.6156582 ]]]]
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

/*
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
*/

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

__global__ 
void GPU_in_hidden(double *input, double *output, double *kernel, double bias[343] ){
	// kernel 676 * 343
	int t= threadIdx.x;
	int stride = blockDim.x;
	int N = 343;

	for (int thd = t; thd < N; thd += stride)
	{
		
		output[thd] = 0.0;
		for (int j = 0; j < 26*26; j++)
		{
			output[thd] += input[j] * kernel[j* 343 + thd];
		}
			
		//bias
		
		
		output[thd] += bias[thd];
		
		//RELU 
		
		output[thd] = output[thd] > 0.0?output[thd]:0.0; 
		
	}
	

	

}

void Layer::dense(double *input, double kernel[343][10], double bias[10]){
	
	for (int i = 0; i < 10; i++)
	{
		this->data1D[i] = 0.0;
	}

	//dot
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 343; j++)
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

__global__
void GPU_dense(double *input, double *output, double *kernel, double bias[10] ){
	// kernel 343 * 10
	int t= threadIdx.x;
	int stride = blockDim.x;
	int N = 10;
	__shared__ double max;
	__shared__ double sum;

	for (int thd = t; thd < N; thd += stride)
	{
		
		output[thd] = 0.0;

		for (int j = 0; j < 343; j++)
		{
			output[thd] += input[j] * kernel[j * 10 + thd];
		}
			
		//bias
		
		output[thd] += bias[thd];
		__syncthreads();

		// printf("thd = %d , %lf \n",thd, output[thd]);
		
		//softmax as activate function
		if (thd == N-1)
		{
			sum = 0.0;
			max = output[0];
			for (int i = 0; i < 10; i++)
			{
				// printf("%lf ",output[i]);
				if (output[i] > max)
				{
					max = output[i];
				}
				
				
			}
			// printf("\n");
			// printf("max = %lf \n",max);

			for (int i = 0; i < 10; i++)
			{
				sum += exp(output[i] - max);	
			}
			
			// printf("sum = %lf \n",sum);
				
		}
		__syncthreads();
		
		output[thd] =  ceil( ( exp(output[thd] - max)/sum ) * 100) / 100;
		
		// output[thd] = ceil( ( exp(output[thd] - max)/sum ) * 100) / 100;
		
		
	}
}

__global__
void GPU_partial(double* input, double* output){
	//kernel 676 * 343
	int t= threadIdx.x;
	int stride = blockDim.x;
	int N = 343;

	for (int thd = t; thd < N; thd += stride){
		output[thd] = 0.0;
		for (int j = 0; j < 676; j++)
		{
			output[thd] += input[j * 343 + thd];
		}
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