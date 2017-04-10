#include "cuda_runtime.h"
#include "nne.cuh"

#include <iostream>
#include <cstdlib>
#include <ctime>

int main()
{
	Layer inputLayer(2, 2);
	Layer hiddenLayer(3, 2);
	Layer outputLayer(2, 3);
	std::vector<float> inputList;
	std::vector<float> outputList;
	float EMS;
	srand((unsigned int)time(NULL));
	inputList.push_back(1);
	inputList.push_back(1);
	outputList.push_back(0);
	outputList.push_back(0);
	cudaSetDevice(0);

	for (int i = 0; i < 20; i++) {
		inputLayer.forwardCal(inputList);
		hiddenLayer.forwardCal(inputLayer);
		outputLayer.forwardCal(hiddenLayer);
		EMS = outputLayer.getGrad(outputList);
		std::cout << EMS << std::endl;
		hiddenLayer.getGrad(outputLayer);
		inputLayer.getGrad(hiddenLayer);
		inputLayer.learnWeight(inputList, 0.4);
		hiddenLayer.learnWeight(inputLayer, 0.4);
		outputLayer.learnWeight(hiddenLayer, 0.4);
	}

    cudaDeviceReset();
    return 0;
}
