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

// Define layers of CNN
static Layer train_input = Layer(1, 28, 28);
static Layer conved = Layer(1, 27, 27);
static Layer maxpooled = Layer(1, 26, 26);
static Layer flattenned = Layer(1, 1, 26*26);
static Layer densed = Layer(1, 1, 10);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

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
			if (data[i][j] != 0.0)
			{
				printf("1 ");
			}
			else{
				printf("0 ");
			}
			
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

int main(int argc, const  char **argv)
{
	loaddata();
	//printimg(train_set[0].data);
	train_one_hot(train_set, train_onehot);
	test_one_hot(test_set, test_onehot);
	train_input.readInput(train_set[0].data);
	conved = train_input.conv2D();
	conved.printData();
	
	//forward_pass(train_set[0].data);
	//learn();
	//test();

	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28])
{
	float input[28][28];
	
	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	clock_t start, end;
	start = clock();

	// 

	

	end = clock();

	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
	clock_t start, end;

	start = clock();

	

	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

static void learn()
{
	
	int epoch = 75;
	float err;
	double time_taken = 0.0;

	fprintf(stdout ,"Learning \n");

	while (epoch > 0) {	
		err = 0.0f;
		

		for (int i = 0; i < train_cnt; ++i) {
			

			//time_taken += conv2D(train_set[i].data);

			
			//checkCuda( cudaDeviceSynchronize() );

			

			time_taken += back_pass();
			
		}

		err /= train_cnt;
		printf("error: %e, time_on_gpu: %lf \n", err, time_taken);

		if (err <= epsilon ) {
			printf("error is less than the epsilon \n");
			break;
		}

		epoch--;
	}

	fprintf(stdout, "\n Time - %lf\n", time_taken);
	
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
	float res[10];

	forward_pass(data);

	unsigned int max = 0;

	//cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}



// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		int classify_label = classify(test_set[i].data);
		if (classify_label != test_set[i].label) {
			//printf("%d label = %d , predict = %d \n", i, test_set[i].label, classify_label);
			//printimg(test_set[i].data);
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n", double(error) / double(test_cnt) * 100.0);
}
