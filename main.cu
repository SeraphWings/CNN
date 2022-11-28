#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <time.h>
#include <cassert>
#include <math.h>
#include <stdio.h>   
#include <stdlib.h> 

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

int train_onehot[60000][10];
int test_onehot[10000][10];
double input_kernel[26*26*6][2033];
double output_kernel[2033][10];
double input_bias[2033];
double output_bias[10];
double error;
double L_rate = 1.0e-5f;

// Define layers of CNN
static Layer train_input = Layer(1, 28, 28);
static Layer conved = Layer(6, 27, 27);
static Layer maxpooled = Layer(6, 26, 26);
static Layer flattenned = Layer(1, 1, 26*26*6);
static Layer hidden = Layer(1, 1, 2033);
static Layer densed = Layer(1, 1, 10);

static void learn();
static unsigned int classify(double data[28][28], int cnt);
double test_on_train();
double test_on_test();
static double forward_pass(double data[28][28], int cnt);
static double back_pass(int cnt);


static inline void loaddata()
{
	mnist_load("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("./data/t10k-images.idx3-ubyte", "./data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);

}

/*
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}
*/

static void printimg(double data[28][28]){
	for (int i = 0; i < 28; ++i)
	{
		for (int j = 0; j < 28; ++j)
		{
			printf("%.2lf ", data[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void train_one_hot(mnist_data *data, int output[60000][10]){
	for(int i = 0;i < 60000;i++){
		for(int j = 0;j < 10;j++){
			if(data[i].label == j){
				output[i][j] = 1;
			}
			else{
				output[i][j] = 0;
			}
			
		}
	}
}

void test_one_hot(mnist_data *data, int output[10000][10]){
	for(int i = 0;i < 10000;i++){
		for(int j = 0;j < 10;j++){
			if(data[i].label == j){
				output[i][j] = 1;
			}
			else{
				output[i][j] = 0;
			}
			
		}
	}
}


void generateKernel(){

	for (int i = 0; i < 26 * 26 * 6; i++)
	{
		for (int j = 0; j < 2033; j++)
		{
			input_kernel[i][j] = ( ( rand() + -1 * rand() )%10) * 0.1;

		}

	}

	for (int i = 0; i < 2033; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			output_kernel[i][j] = ( ( rand() + -1 * rand() )%10) * 0.1;

		}

	}

}

void generateBias(){

		for (int i = 0; i < 2033; i++)
		{
			input_bias[i] = ( ( rand() + -1 * rand() )%10) *0.1;

		}

		for (int i = 0; i < 10; i++)
		{
			output_bias[i] = ( ( rand() + -1 * rand() )%10) *0.1;

		}

}



double crossEntropy(int cnt){
	//printf("cross entropy\n");
	double output = 0.0;
	for (int i = 0; i < 10; i++)
	{
		//printf("%.2lf ", densed.data1D[i]);
		if(densed.data1D[i] > 0){
			
			output += log( densed.data1D[i] ) * train_onehot[cnt][i];
		}
		else if(densed.data1D[i] == 0.0){
			output += log( 1.0e-4f ) * train_onehot[cnt][i];
		}
		else{
			printf("%d %d log() error\n", cnt, i);
		}

	}
	//printf("\n");

	output *= -1.0;
	output /= 10.0;

	return output;
	
}

void test_flatten(){
	for (int i = 0; i < maxpooled.C; i++)
	{
		for (int j = 0; j < maxpooled.H; j++)
		{
			for (int k = 0; k < maxpooled.W; k++)
			{
				flattenned.data1D[i * maxpooled.H * maxpooled.W + j * maxpooled.W + k] = maxpooled.data3D[i][j][k];
			}
			
			
		}
		
	}
}

int main(int argc, const  char **argv)
{
	loaddata();
	//printimg(train_set[0].data);
	train_one_hot(train_set, train_onehot);
	test_one_hot(test_set, test_onehot);
	srand (time(NULL));
	generateKernel();
	generateBias();

	printf("----- forward passing takes %lf milliseconds\n", forward_pass(train_set[0].data, 0));
	printf("----- backward passing takes %lf milliseconds\n", back_pass(0));
	// printf("----- forward passing takes %lf milliseconds\n", forward_pass(train_set[3].data, 3));
	// printf("----- backward passing takes %lf milliseconds\n", back_pass(3));
	// printimg(test_set[0].data);
	
	// learn();
	
	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28], int cnt){

	clock_t start, end, tmp;
	start = clock();
	
	tmp = clock();
	// printf("origin\n");
	train_input.readInput(train_set[cnt].data);
	// train_input.printData();
	printf("data read in takes %lf milliseconds\n", (double)(clock() - tmp) / ( CLOCKS_PER_SEC/1000));
	
	tmp = clock();
	// printf("convolution\n");
	conved.conv2D(train_input.data2D);
	// conved.printData();
	printf("conv2D layer takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	
	tmp = clock();
	// printf("maxpooling\n");
	maxpooled.maxPooling(conved.data3D);
	// maxpooled.printData();
	printf("maxpooling layer takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	
	tmp = clock();
	// printf("flatten\n");
	test_flatten();
	//flattenned.flatten(maxpooled.data2D);
	// flattenned.printData();
	printf("flatten layer takes %lf millieconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));


	int dummyInt = 1;
	int* dummy;
	if ( cudaMalloc((void**)&dummy, sizeof(int) * 1)  != cudaSuccess) printf("dummy error\n");
	if ( cudaMemcpy(dummy, &dummyInt, sizeof(int) * 1, cudaMemcpyHostToDevice) != cudaSuccess) printf("dummy cpy error\n");
	cudaFree(dummy);
	
	tmp = clock();
	// printf("hidden\n");
	// hidden.in_hidden(flattenned.data1D, input_kernel, input_bias);
	// printf("into hidden layer - calculation takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));

	
	
	double* device_flatten;
	if ( cudaMalloc((void**)&device_flatten, sizeof(double) * 676*6)  != cudaSuccess) printf("device_flatten error\n");
	if ( cudaMemcpy(device_flatten, flattenned.data1D, sizeof(double) * 676*6, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_flatten cpy error\n");
	double* device_hidden;
	if ( cudaMalloc((void**)&device_hidden, sizeof(double) * 2033) != cudaSuccess) printf("device_hidden error\n");

	// double* host_input_kernel;
	// cudaMallocHost((void**)&host_input_kernel, sizeof(double) * 676 * 6 * 2033, cudaHostAllocDefault);
	// memcpy( &host_input_kernel, &input_kernel,  sizeof(double) * 676 * 6 * 2033);

	double* device_input_kernel;
	if ( cudaMalloc((void**)&device_input_kernel, sizeof(double) * 676 * 6 * 2033) != cudaSuccess) printf("device_input_kernel error\n");
	if ( cudaMemcpy(device_input_kernel, input_kernel, sizeof(double) * 676 * 6 * 2033, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_input_kernel cpy error\n");
	
	double* device_input_bias;
	if ( cudaMalloc((void**)&device_input_bias, sizeof(double) * 2033) != cudaSuccess) printf("device_input_bias error\n");
	if ( cudaMemcpy(device_input_bias, input_bias, sizeof(double) * 2033, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_input_bias cpy error\n");
	printf("into hidden layer - memory to GPU takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	
	tmp = clock();
	// GPU_in_hidden<<<2048,2048>>>(device_flatten, device_hidden, host_input_kernel, device_input_bias);
	GPU_in_hidden<<<1024,1024>>>(device_flatten, device_hidden, device_input_kernel, device_input_bias);
	cudaDeviceSynchronize();
	printf("into hidden layer - calculation takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	
	tmp = clock();
	if ( cudaMemcpy(hidden.data1D, device_hidden, sizeof(double) * 2033, cudaMemcpyDeviceToHost) != cudaSuccess) printf("device_hidden cpy back error\n");
	printf("into hidden layer - memory to CPU takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	
	cudaFree(device_flatten);
	cudaFree(device_input_kernel);
	// cudaFree(host_input_kernel);
	cudaFree(device_input_bias);
	// hidden.printData();
	// printf("into hidden layer takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	

	tmp = clock();
	// printf("densed\n");
	// densed.dense(hidden.data1D, output_kernel, output_bias);
	// printf("dense layer - calculation takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	
	// double* device_hidden;
	// if ( cudaMalloc((void**)&device_hidden, sizeof(double) * 2033) != cudaSuccess) printf("device_hidden error\n");
	// if ( cudaMemcpy(device_hidden, hidden.data1D, sizeof(double) * 2033, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_output_kernel cpy error\n");
	double* device_densed;
	if ( cudaMalloc((void**)&device_densed, sizeof(double) * 10) != cudaSuccess) printf("device_densed error\n");
	double* device_output_kernel;
	if ( cudaMalloc((void**)&device_output_kernel, sizeof(double) * 10 * 2033) != cudaSuccess) printf("device_output_kernel error\n");
	if ( cudaMemcpy(device_output_kernel, output_kernel, sizeof(double) * 10 * 2033, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_output_kernel cpy error\n");
	double* device_output_bias;
	if ( cudaMalloc((void**)&device_output_bias, sizeof(double) * 10) != cudaSuccess) printf("device_output_bias error\n");
	if ( cudaMemcpy(device_output_bias, output_bias, sizeof(double) * 10, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_output_bias cpy error\n");
	printf("dense layer - memory to GPU takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));

	tmp = clock();
	GPU_dense<<<1,10>>>(device_hidden, device_densed, device_output_kernel, device_output_bias);
	cudaDeviceSynchronize();
	printf("dense layer - calculation takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	tmp = clock();
	if ( cudaMemcpy(densed.data1D, device_densed, sizeof(double) * 10, cudaMemcpyDeviceToHost) != cudaSuccess) printf("device_dense cpy back error\n");
	
	cudaFree(device_hidden);
	cudaFree(device_densed);
	cudaFree(device_output_kernel);
	cudaFree(device_output_bias);
	// if(cnt % 1000 == 0) densed.printData();
	// densed.printData();
	printf("dense layer - memory to CPU takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	

	tmp = clock();
	error = crossEntropy(cnt);
	// if(cnt % 10000 == 0) printf("%d label = %d , predict = %d \n", cnt, train_set[cnt].label, classify(train_set[cnt].data, cnt));
	//if(cnt < 5) printf("cnt = %d \t error = %lf\n", cnt, error);
	//if(cnt % 1000 == 0) printf("cnt = %d \t error = %lf\n", cnt, error);
	//if(cnt % 10000 == 0) printf("cnt = %d \t error = %lf\n", cnt, error);
	printf("error calculation takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));

	end = clock();

	return ((double) (end - start)) / (CLOCKS_PER_SEC/1000);
}

// Back propagation to update weights
static double back_pass(int cnt)
{
	//printf("back propagation\n");
	clock_t start, end,tmp;

	start = clock();


	tmp = clock();
	//output delta
	double output_delta[10];
	for (int i = 0; i < 10; i++)
	{
		output_delta[i] = densed.data1D[i] - (double)train_onehot[cnt][i];
		
	}
	printf("output delta takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));

	// printf("out delta\n");
	// for (int i = 0; i < 10; i++)
	// {
	// 	printf("%.2lf ", output_delta[i]);

	// }
	// printf("\n");

	//hidden delta
	tmp = clock();
	double hidden_delta[2033];
	double pre_act[2033];
	for (int i = 0; i < 2033; i++)
	{
		pre_act[i] = 0.0;
		for (int j = 0; j < 10; j++)
		{
			pre_act[i] += output_delta[j] * output_kernel[i][j];
		}
		
	}
	printf("pre_act calculation takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	// printf("pre_act\n");
	// for (int i = 0; i < 2033; i++)
	// {
	// 	printf("%.2lf ", pre_act[i]);

	// }
	// printf("\n");



	tmp = clock();
		/*
	for (int i = 0; i < 343; i++)
	{
		partial[i] = 0.0;
		for (int j = 0; j < 676; j++)
		{
			partial[i] += input_kernel[j][i];
		}
			
	}
	*/

	double partial[2033];
	double* device_partial;
	if (cudaMalloc((void**)&device_partial, sizeof(double)*2033) != cudaSuccess) printf("device_partial error\n");
	double* device_input_kernel;
	if (cudaMalloc((void**)&device_input_kernel,sizeof(double) * 676 * 6 * 2033) != cudaSuccess) printf("device_input_kernel error\n");
	if ( cudaMemcpy(device_input_kernel, input_kernel, sizeof(double)* 676 * 6 * 2033, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_input_kernel cpy error\n");
	GPU_partial<<<1024,1024>>>(device_input_kernel, device_partial);
	cudaDeviceSynchronize();
	if ( cudaMemcpy(partial, device_partial, sizeof(double)* 2033, cudaMemcpyDeviceToHost) != cudaSuccess) printf("device_input_kernel cpy back error\n");
	

	cudaFree(device_partial);
	cudaFree(device_input_kernel);
	printf("patrial differential takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));


	tmp = clock();
	for (int i = 0; i < 2033; i++)
	{
		
		hidden_delta[i] = pre_act[i] * partial[i];
		
	}
	printf("hidden_delta takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));


	// printf("hidden delta\n");
	// for (int i = 0; i < 2033; i++)
	// {
	// 	printf("%lf ", hidden_delta[i]);
		
	// }
	// printf("\n");
	

	/*
	//input delta
	double input_delta[26*26];
	for (int i = 0; i < 26*26; i++)
	{
		input_delta[i] = 0.0;
		for (int j = 0; j < 343; j++)
		{
			input_delta[i] += flattenned.data1D[i] * hidden_delta[j] * input_kernel[i][j];
		}
		
	}
	*/
	
	//output kernel&bias update
	tmp = clock();
	for (int i = 0; i < 2033; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			output_kernel[i][j] -= L_rate * hidden.data1D[j] * output_delta[j];
		}

	}
	printf("output_kernel update takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	// printf("output kernel\n");
	// for (int i = 0; i < 343; i++)
	// {
	// 	for (int j = 0; j < 10; j++)
	// 	{
	// 		printf("%lf ", output_kernel[i][j]);
	// 	}

	// }
	// printf("\n");

	tmp = clock();
	for (int j = 0; j < 10; j++)
	{
		output_bias[j] -=   L_rate *  output_delta[j];
	}
	printf("output_bias update takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	// printf("output bias\n");
	// for (int i = 0; i < 10; i++)
	// {
	// 	printf("%lf ", output_bias[i]);
	// }
	// printf("\n");

	// printf("input kernel before \n");
	// for (int i = 0; i < 26*26; i++)
	// {
		
	// 	printf("%lf ", input_kernel[i][0]);
	// }
	// printf("\n");

	//input kernel&bias update
	tmp = clock();
	for (int i = 0; i < 26 * 26 * 6; i++)
	{
		for (int j = 0; j < 2033; j++)
		{
			input_kernel[i][j] = input_kernel[i][j] - L_rate * flattenned.data1D[i] * hidden_delta[j];
		}

	}
	printf("input_kernel update takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));
	// printf("input kernel\n");
	// for (int i = 0; i < 26*26 * 6; i++)
	// {
		
	// 	printf("%lf ", input_kernel[i][123]);
	// }
	// printf("\n");

	tmp = clock();
	for (int j = 0; j < 2033; j++)
	{
		input_bias[j] -=   L_rate *  hidden_delta[j];
	}
	printf("input_bias update takes %lf milliseconds\n", (double) (clock() - tmp) / (CLOCKS_PER_SEC/1000));

	end = clock();
	return ((double) (end - start)) / (CLOCKS_PER_SEC/1000);
}

static void learn()
{
	
	int epoch = 3;
	double time_taken = 0.0;
	int patience = 2;
	double patience_factor = 0.2;
	int patience_idx = 0;
	bool *patience_test = new bool(patience);
	double last_epoch_err;
	train_cnt = 10000;
	
	fprintf(stdout ,"Learning \n");

	for (int epoch_cnt = 0; epoch_cnt < epoch; epoch_cnt++)
	{
		double epoch_err = 0.0;
		int train_idx;
		printf("epoch %d\n", epoch_cnt);
		for (int i = 0; i < train_cnt; i++) {
			
			//printf("forward passing\n");
			train_idx = rand()%train_cnt;
			time_taken += forward_pass(train_set[train_idx].data, train_idx);
			time_taken += back_pass(train_idx);
			epoch_err += error;
			if(i % 1000 == 0) printf("i = %d \t idx = %d \t error: %lf\n", i,train_idx, epoch_err/(i+1));
			if(i % 1000 == 0) printf("label = %d , predict = %d \n", train_set[train_idx].label, classify(train_set[train_idx].data, train_idx));
			//if(i % 1000 == 0) printf("error: %lf\n", error);

		}

		printf("epoch %d \t error: %lf \t time_on_gpu: %lf \n",epoch_cnt, epoch_err/train_cnt, time_taken);

		if(epoch_cnt == 0) last_epoch_err = epoch_err/train_cnt;
		patience_test[ (patience_idx+1) % patience ] = (last_epoch_err - epoch_err/train_cnt) >= 0? false:true;
		last_epoch_err = epoch_err/train_cnt;
		if (epoch_cnt > 1 && patience_test[0] && patience_test[1]) L_rate *= patience_factor;
	
		test_on_train();
		//test_on_test();
		printf("-----\n");

	}

	fprintf(stdout, "\nTime - %lf\n", time_taken);
	
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28], int cnt)
{
	double res[10];
	unsigned int max = 0;

	//cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; ++i) {
		res[i] = densed.data1D[i];
	}
	for (int i = 0; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}



// Perform forward propagation of test data
double test_on_test()
{
	int test_acc = 0;

	for (int i = 0; i < test_cnt; ++i) {
		forward_pass(test_set[i].data, i);
		int classify_label = classify(test_set[i].data, i);
		if (classify_label == test_set[i].label) {
			//printf("%d label = %d , predict = %d \n", i, test_set[i].label, classify_label);
			//printimg(test_set[i].data);
			test_acc++;
		}
		

	}

	fprintf(stdout, "test on test accuracy: %.2lf%%\n", double(test_acc) / double(test_cnt) * 100.0);
	return double(test_acc) / double(test_cnt) * 100.0;
}

// Perform forward propagation of test data
double test_on_train()
{
	int test_acc = 0;

	for (int i = 0; i < train_cnt; ++i) {
		forward_pass(train_set[i].data, i);
		int classify_label = classify(train_set[i].data, i);
		if (classify_label == train_set[i].label) {
			//printf("%d label = %d , predict = %d \n", i, test_set[i].label, classify_label);
			//printimg(test_set[i].data);
			test_acc++;
		}
		

	}

	fprintf(stdout, "test on train accuracy: %.2lf%%\n", double(test_acc) / double(train_cnt) * 100.0);
	return double(test_acc) / double(train_cnt) * 100.0;
}
