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
double input_kernel[26*26][343];
double output_kernel[343][10];
double input_bias[343];
double output_bias[10];
double error;
double L_rate = 1.0e-5f;

// Define layers of CNN
static Layer train_input = Layer(1, 28, 28);
static Layer conved = Layer(1, 27, 27);
static Layer maxpooled = Layer(1, 26, 26);
static Layer flattenned = Layer(1, 1, 26*26);
static Layer hidden = Layer(1, 1, 343);
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

	for (int i = 0; i < 26*26; i++)
	{
		for (int j = 0; j < 343; j++)
		{
			input_kernel[i][j] = ( ( rand() + -1 * rand() )%10) * 0.1;

		}

	}

	for (int i = 0; i < 343; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			output_kernel[i][j] = ( ( rand() + -1 * rand() )%10) * 0.1;

		}

	}

}

void generateBias(){

		for (int i = 0; i < 343; i++)
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
	for (int i = 0; i < maxpooled.H; i++)
	{
		for (int j = 0; j <maxpooled.W; j++)
		{
			flattenned.data1D[i * maxpooled.W + j] = maxpooled.data2D[i][j];
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

	/*
	printf("origin\n");
	train_input.readInput(train_set[0].data);
	train_input.printData();
	
	printf("convolution\n");
	conved.conv2D(train_input.data2D);
	conved.printData();
	
	printf("maxpooling\n");
	maxpooled.maxPooling(conved.data2D);
	maxpooled.printData();
	
	printf("flatten\n");
	test_flatten();
	//flattenned.flatten(maxpooled.data2D);
	flattenned.printData();

	printf("hidden\n");
	//hidden.in_hidden(flattenned.data1D, input_kernel, input_bias);
	double* device_flatten;
	if ( cudaMalloc((void**)&device_flatten, sizeof(double) * 676)  != cudaSuccess) printf("device_flatten error\n");
	if ( cudaMemcpy(device_flatten, flattenned.data1D, sizeof(double) * 676, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_flatten cpy error\n");
	double* device_hidden;
	if ( cudaMalloc((void**)&device_hidden, sizeof(double) * 343) != cudaSuccess) printf("device_hidden error\n");
	double* device_input_kernel;
	if ( cudaMalloc((void**)&device_input_kernel, sizeof(double) * 676 * 343) != cudaSuccess) printf("device_input_kernel error\n");
	if ( cudaMemcpy(device_input_kernel, input_kernel, sizeof(double) * 676 * 343, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_input_kernel cpy error\n");
	double* device_input_bias;
	if ( cudaMalloc((void**)&device_input_bias, sizeof(double) * 343) != cudaSuccess) printf("device_input_bias error\n");
	if ( cudaMemcpy(device_input_bias, input_bias, sizeof(double) * 343, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_input_bias cpy error\n");

	
	GPU_in_hidden<<<1,1>>>(device_flatten, device_hidden, device_input_kernel, device_input_bias);
	if ( cudaMemcpy(hidden.data1D, device_hidden, sizeof(double) * 343, cudaMemcpyDeviceToHost) != cudaSuccess) printf("device_hidden cpy back error\n");
	cudaDeviceSynchronize();
	cudaFree(device_flatten);
	cudaFree(device_input_kernel);
	cudaFree(device_input_bias);
	cudaFree(device_hidden);
	hidden.printData();
	
	printf("densed\n");
	densed.dense(hidden.data1D, output_kernel, output_bias);
	densed.printData();
	
	error = crossEntropy(0);
	*/

	// printf("%lf \n", forward_pass(train_set[0].data, 0));
	// back_pass(0);
	// // test_on_train();
	// printf("%lf \n", forward_pass(train_set[3].data, 3));
	// back_pass(3);
	// printimg(test_set[0].data);
	
	learn();
	
	/*
	learn();
	//test();
	*/
	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28], int cnt){

	clock_t start, end;
	start = clock();

	// printf("origin\n");
	train_input.readInput(train_set[cnt].data);
	// train_input.printData();
	
	// printf("convolution\n");
	conved.conv2D(train_input.data2D);
	// conved.printData();
	
	// printf("maxpooling\n");
	maxpooled.maxPooling(conved.data2D);
	// maxpooled.printData();
	
	// printf("flatten\n");
	test_flatten();
	//flattenned.flatten(maxpooled.data2D);
	// flattenned.printData();

	// printf("hidden\n");
	// hidden.in_hidden(flattenned.data1D, input_kernel, input_bias);
	double* device_flatten;
	if ( cudaMalloc((void**)&device_flatten, sizeof(double) * 676)  != cudaSuccess) printf("device_flatten error\n");
	if ( cudaMemcpy(device_flatten, flattenned.data1D, sizeof(double) * 676, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_flatten cpy error\n");
	double* device_hidden;
	if ( cudaMalloc((void**)&device_hidden, sizeof(double) * 343) != cudaSuccess) printf("device_hidden error\n");
	double* device_input_kernel;
	if ( cudaMalloc((void**)&device_input_kernel, sizeof(double) * 676 * 343) != cudaSuccess) printf("device_input_kernel error\n");
	if ( cudaMemcpy(device_input_kernel, input_kernel, sizeof(double) * 676 * 343, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_input_kernel cpy error\n");
	double* device_input_bias;
	if ( cudaMalloc((void**)&device_input_bias, sizeof(double) * 343) != cudaSuccess) printf("device_input_bias error\n");
	if ( cudaMemcpy(device_input_bias, input_bias, sizeof(double) * 343, cudaMemcpyHostToDevice) != cudaSuccess) printf("device_input_bias cpy error\n");

	
	GPU_in_hidden<<<1,64>>>(device_flatten, device_hidden, device_input_kernel, device_input_bias);
	if ( cudaMemcpy(hidden.data1D, device_hidden, sizeof(double) * 343, cudaMemcpyDeviceToHost) != cudaSuccess) printf("device_hidden cpy back error\n");
	cudaDeviceSynchronize();
	cudaFree(device_flatten);
	cudaFree(device_input_kernel);
	cudaFree(device_input_bias);
	cudaFree(device_hidden);
	// hidden.printData();
	
	// printf("densed\n");
	densed.dense(hidden.data1D, output_kernel, output_bias);
	// if(cnt % 1000 == 0) densed.printData();
	// densed.printData();
	
	error = crossEntropy(cnt);
	// if(cnt % 10000 == 0) printf("%d label = %d , predict = %d \n", cnt, train_set[cnt].label, classify(train_set[cnt].data, cnt));
	//if(cnt < 5) printf("cnt = %d \t error = %lf\n", cnt, error);
	//if(cnt % 1000 == 0) printf("cnt = %d \t error = %lf\n", cnt, error);
	//if(cnt % 10000 == 0) printf("cnt = %d \t error = %lf\n", cnt, error);

	end = clock();

	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass(int cnt)
{
	//printf("back propagation\n");
	clock_t start, end;

	start = clock();

	//output delta
	double output_delta[10];
	for (int i = 0; i < 10; i++)
	{
		output_delta[i] = densed.data1D[i] - (double)train_onehot[cnt][i];
		
	}

	// printf("out delta\n");
	// for (int i = 0; i < 10; i++)
	// {
	// 	printf("%.2lf ", output_delta[i]);

	// }
	// printf("\n");

	//hidden delta
	double hidden_delta[343];
	double pre_act[343];
	for (int i = 0; i < 343; i++)
	{
		pre_act[i] = 0.0;
		for (int j = 0; j < 10; j++)
		{
			pre_act[i] += output_delta[j] * output_kernel[i][j];
		}
		
	}

	double partial[343];
	for (int i = 0; i < 343; i++)
	{
		partial[i] = 0.0;
		for (int j = 0; j < 676; j++)
		{
			partial[i] += input_kernel[j][i];
		}
			
	}
	
	for (int i = 0; i < 343; i++)
	{
		
		hidden_delta[i] = pre_act[i] * partial[i];
		
	}



	// printf("hidden delta\n");
	// for (int i = 0; i < 343; i++)
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
	for (int i = 0; i < 343; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			output_kernel[i][j] -= L_rate * hidden.data1D[j] * output_delta[j];
		}

	}
	// printf("output kernel\n");
	// for (int i = 0; i < 343; i++)
	// {
	// 	for (int j = 0; j < 10; j++)
	// 	{
	// 		printf("%lf ", output_kernel[i][j]);
	// 	}

	// }
	// printf("\n");

	for (int j = 0; j < 10; j++)
	{
		output_bias[j] -=   L_rate *  output_delta[j];
	}

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
	for (int i = 0; i < 26*26; i++)
	{
		for (int j = 0; j < 343; j++)
		{
			input_kernel[i][j] = input_kernel[i][j] - L_rate * flattenned.data1D[i] * hidden_delta[j];
		}

	}

	// printf("input kernel\n");
	// for (int i = 0; i < 26*26; i++)
	// {
		
	// 	printf("%lf ", input_kernel[i][0]);
	// }
	// printf("\n");

	for (int j = 0; j < 343; j++)
	{
		input_bias[j] -=   L_rate *  hidden_delta[j];
	}


	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

static void learn()
{
	
	int epoch = 3;
	double time_taken = 0.0;
	

	fprintf(stdout ,"Learning \n");

	while (epoch > 0) {	
		double epoch_err = 0.0;
		int train_idx;
		printf("epoch %d\n", epoch);
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
		
		printf("epoch %d \t error: %lf \t time_on_gpu: %lf \n",epoch, epoch_err/train_cnt, time_taken);
		test_on_train();
		test_on_test();
		printf("-----\n");
		epoch--;
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
