#include "nne.cuh"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <cstdlib>

__global__ void nodeCal(float* inList, float* wList, float* outList, int inputNum);
__global__ void nodeLog(float* outputList);
__global__ void nodeLearn(float input, float learnConst, float* wList, float* outList);

Node::Node() : output(0), input(0), localGrad(0) {
	inputWeightList.push_back(0);
}

Node::Node(int inputNum) : output(0), input(0), localGrad(0) {
	inputWeightList.push_back(0);
	for (int i = 0; i < inputNum; i++) {
		inputWeightList.push_back((float)rand() / RAND_MAX);
	}
}

Node::Node(std::vector<float>& inputWeightList, int nodeIndex, int inputWeightLength) : output(0), input(0), localGrad(0) {
	int offset = nodeIndex * inputWeightLength;
	inputWeightList.push_back(0);
	for (int i = 0; i < inputWeightLength; i++) {
		inputWeightList.push_back(inputWeightList[offset + i]);
	}
}

Node::~Node() {}

Layer::Layer(){}

Layer::Layer(int nodeListLength, int inputWeightLength) {
	Node* newNode;
	for (int i = 0; i < nodeListLength; i++) {
		newNode = new Node(inputWeightLength);
		nodeList.push_back(newNode);
	}
}

Layer::Layer(std::vector<float>& inputWeightList, int nodeListLength, int inputWeightLength) {
	Node* newNode;
	for (int i = 0; i < nodeListLength; i++) {
		newNode = new Node(inputWeightList, i, inputWeightLength);
		nodeList.push_back(newNode);
	}
}

Layer::~Layer() {
	int length = nodeList.size();
	for (int i = 0; i < length; i++) {
		delete nodeList[i];
	}
}

void Layer::forwardCal(std::vector<float>& inputList) {
	int inputNum = inputList.size();
	int outputNum = nodeList.size();
	std::vector<float> weightList;
	float* outputList = new float[outputNum];
	float *dInputList, *dWeightList, *dOutputList;

	cudaMalloc(&dInputList, inputNum * sizeof(float));
	cudaMalloc(&dWeightList, inputNum * outputNum * sizeof(float));
	cudaMalloc(&dOutputList, outputNum * sizeof(float));

	cudaMemcpy(dInputList, inputList.data(), inputNum * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < outputNum; i++) {
		weightList.insert(weightList.end(), nodeList[i]->inputWeightList.begin(), nodeList[i]->inputWeightList.end());
	}

	cudaMemcpy(dWeightList, weightList.data(), inputNum * outputNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dOutputList, outputList, outputNum * sizeof(float), cudaMemcpyHostToDevice);

	nodeCal <<<outputNum, inputNum, sizeof(float) * inputNum>>> (dInputList, dWeightList, dOutputList, inputNum);
	cudaMemcpy(outputList, dOutputList, outputNum * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < outputNum; i++) {
		nodeList[i]->input = outputList[i];
	}

	nodeLog <<<1, outputNum >>> (dOutputList);
	cudaMemcpy(outputList, dOutputList, outputNum * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dInputList);
	cudaFree(dWeightList);
	cudaFree(dOutputList);
	for (int i = 0; i < outputNum; i++) {
		nodeList[i]->output = outputList[i];
	}
	delete outputList;
}

void Layer::forwardCal(Layer& bLayer){
	std::vector<Node*> &bNodeList = bLayer.nodeList;
	int inputNum = bNodeList.size();
	int outputNum = nodeList.size();
	std::vector<float> inputList;
	std::vector<float> weightList;
	float* outputList = new float[outputNum];
	float *dInputList, *dWeightList, *dOutputList;
	
	cudaMalloc(&dInputList, inputNum * sizeof(float));
	cudaMalloc(&dWeightList, inputNum * outputNum * sizeof(float));
	cudaMalloc(&dOutputList, outputNum * sizeof(float));
	
	inputList.push_back(1);
	for (int i = 0; i < inputNum; i++) {
		inputList.push_back((*bNodeList[i]).output);
	}
	inputNum++;

	cudaMemcpy(dInputList, inputList.data(), inputNum * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < outputNum; i++) {
		weightList.insert(weightList.end(), nodeList[i]->inputWeightList.begin(), nodeList[i]->inputWeightList.end());
	}

	cudaMemcpy(dWeightList, weightList.data(), inputNum * outputNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dOutputList, outputList, outputNum * sizeof(float), cudaMemcpyHostToDevice);

	nodeCal <<<outputNum, inputNum, sizeof(float) * inputNum >>> (dInputList, dWeightList, dOutputList, inputNum);
	cudaMemcpy(outputList, dOutputList, outputNum * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < outputNum; i++) {
		nodeList[i]->input = outputList[i];
	}
	nodeLog <<<1, outputNum >>> (dOutputList);
	cudaMemcpy(outputList, dOutputList, outputNum * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dInputList);
	cudaFree(dWeightList);
	cudaFree(dOutputList);
	for (int i = 0; i < outputNum; i++) {
		nodeList[i]->output = outputList[i];
	}
	delete outputList;
}

void Layer::backPropa(Layer& fLayer, float learningFactor) {
	std::vector<Node*> &bNodeList = bLayer.nodeList;
	int inputNum = bNodeList.size();
	int outputNum = nodeList.size();
	std::vector<float> inputList;
	std::vector<float> weightList;
	float* outputList = new float[outputNum];
	float *dInputList, *dWeightList, *dOutputList;

	cudaMalloc(&dInputList, inputNum * sizeof(float));
	cudaMalloc(&dWeightList, inputNum * outputNum * sizeof(float));
	cudaMalloc(&dOutputList, outputNum * sizeof(float));

	inputList.push_back(1);
	for (int i = 0; i < inputNum; i++) {
		inputList.push_back((*bNodeList[i]).output);
	}
	inputNum++;

	cudaMemcpy(dInputList, inputList.data(), inputNum * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < outputNum; i++) {
		weightList.insert(weightList.end(), nodeList[i]->inputWeightList.begin(), nodeList[i]->inputWeightList.end());
	}

	cudaMemcpy(dWeightList, weightList.data(), inputNum * outputNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dOutputList, outputList, outputNum * sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void nodeCal(float* inputList, float* weightList, float* outputList, int inputNum){
	int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
	float result = 0;
	extern __shared__ float results[];
	results[threadIdx.x] = inputList[threadIdx.x] * weightList[outputIdx];
	__syncthreads();
	for (int i = 0; i < inputNum; i++) {
		result += results[i];
	}
	outputList[blockIdx.x] = result;
	//result = 0;
	//result += subresult;
	//outputList[blockIdx.x] = result;
}

__global__ void nodeLog(float* outputList) {
	outputList[threadIdx.x] = tanh(outputList[threadIdx.x]);
}

__global__ void nodeLearn(float input, float learnConst, float* wList, float* outList) {

}
