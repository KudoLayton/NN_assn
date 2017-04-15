#include "nne.cuh"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <cstdlib>

__global__ void nodeCal(float* inList, float* wList, float* outList);
__global__ void nodeLog(float* outputList, float sigmoidConst);
__global__ void nodeGradCal(float* wList, float* outputList, float* gradList);
__global__ void nodeDelLog(float* inputList, float* gradList, float sigmoidConst);
__global__ void nodeLearn(float *inputList, float *delList, float *weightList, float learningFactor, int inputNum);

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
	inputWeightLength++;
	int offset = nodeIndex * inputWeightLength;
	//inputWeightList.push_back(0);
	for (int i = 0; i < inputWeightLength; i++) {
		this->inputWeightList.push_back(inputWeightList[offset + i]);
	}
}

Node::~Node() {}

Layer::Layer() : sigmoidConst(0.01) {}

Layer::Layer(int nodeListLength, int inputWeightLength, float sigmoidConst){
	Node* newNode;
	this->sigmoidConst = sigmoidConst;
	for (int i = 0; i < nodeListLength; i++) {
		newNode = new Node(inputWeightLength);
		nodeList.push_back(newNode);
	}
}

Layer::Layer(std::vector<float>& inputWeightList, int nodeListLength, int inputWeightLength, float sigmoidConst) {
	Node* newNode;
	this->sigmoidConst = sigmoidConst;
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
	int inputNum = inputList.size() + 1;
	int outputNum = nodeList.size();
	std::vector<float> weightList;
	float* outputList = new float[outputNum];
	float *dInputList, *dWeightList, *dOutputList;

	cudaMalloc(&dInputList, inputNum * sizeof(float));
	cudaMalloc(&dWeightList, inputNum * outputNum * sizeof(float));
	cudaMalloc(&dOutputList, outputNum * sizeof(float));

	inputList.insert(inputList.begin(), 1);

	cudaMemcpy(dInputList, inputList.data(), inputNum * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < outputNum; i++) {
		weightList.insert(weightList.end(), nodeList[i]->inputWeightList.begin(), nodeList[i]->inputWeightList.end());
	}

	cudaMemcpy(dWeightList, weightList.data(), inputNum * outputNum * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(dOutputList, outputList, outputNum * sizeof(float), cudaMemcpyHostToDevice);

	nodeCal <<<outputNum, inputNum, sizeof(float) * inputNum>>> (dInputList, dWeightList, dOutputList);
	cudaMemcpy(outputList, dOutputList, outputNum * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < outputNum; i++) {
		nodeList[i]->input = outputList[i];
	}

	nodeLog <<<1, outputNum >>> (dOutputList, sigmoidConst);
	cudaMemcpy(outputList, dOutputList, outputNum * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dInputList);
	cudaFree(dWeightList);
	cudaFree(dOutputList);
	for (int i = 0; i < outputNum; i++) {
		nodeList[i]->output = outputList[i];
	}
	delete outputList;
	inputList.erase(inputList.begin());
}

void Layer::forwardCal(Layer& bLayer){
	std::vector<Node*> &bNodeList = bLayer.nodeList;
	int inputNum = bNodeList.size();
	int outputNum = nodeList.size();
	std::vector<float> inputList;
	std::vector<float> weightList;
	float* outputList = new float[outputNum];
	float *dInputList, *dWeightList, *dOutputList;
	
	cudaMalloc(&dInputList, (inputNum + 1) * sizeof(float));
	cudaMalloc(&dWeightList, (inputNum + 1) * outputNum * sizeof(float));
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
	//cudaMemcpy(dOutputList, outputList, outputNum * sizeof(float), cudaMemcpyHostToDevice);

	nodeCal <<<outputNum, inputNum, sizeof(float) * inputNum >>> (dInputList, dWeightList, dOutputList);
	cudaMemcpy(outputList, dOutputList, outputNum * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < outputNum; i++) {
		nodeList[i]->input = outputList[i];
	}
	nodeLog <<<1, outputNum >>> (dOutputList, sigmoidConst);
	cudaMemcpy(outputList, dOutputList, outputNum * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dInputList);
	cudaFree(dWeightList);
	cudaFree(dOutputList);
	for (int i = 0; i < outputNum; i++) {
		nodeList[i]->output = outputList[i];
	}
	delete outputList;
}

void Layer::getGrad(Layer& fLayer, int batchNum) {
	int inputNum = nodeList.size();
	int outputNum = fLayer.nodeList.size();
	std::vector<Node*> &fNodeList = fLayer.nodeList;
	std::vector<float> inputList;
	std::vector<float> weightList;
	std::vector<float> outputList;
	float *gradList = new float[inputNum];
	float *dInputList, *dWeightList, *dOutputList, *dGradList;

	cudaMalloc(&dInputList, inputNum * sizeof(float));
	cudaMalloc(&dWeightList, inputNum * outputNum * sizeof(float));
	cudaMalloc(&dOutputList, outputNum * sizeof(float));
	cudaMalloc(&dGradList, inputNum * sizeof(float));

	for (int i = 0; i < inputNum; i++) {
		inputList.push_back(nodeList[i]->input);
	}

	cudaMemcpy(dInputList, inputList.data(), inputNum * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < outputNum; i++) {
		outputList.push_back(fNodeList[i]->localGrad);
		weightList.insert(weightList.end(), ++(fNodeList[i]->inputWeightList.begin()), fNodeList[i]->inputWeightList.end());
	}

	cudaMemcpy(dOutputList, outputList.data(), outputNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dWeightList, weightList.data(), inputNum * outputNum * sizeof(float), cudaMemcpyHostToDevice);

	nodeGradCal <<<inputNum, outputNum, sizeof(float) * outputNum >>> (dWeightList, dOutputList, dGradList);
	cudaMemcpy(gradList, dGradList, inputNum * sizeof(float), cudaMemcpyDeviceToHost);
	nodeDelLog <<<1, inputNum >>> (dInputList, dGradList, sigmoidConst);
	cudaMemcpy(gradList, dGradList, inputNum * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dInputList);
	cudaFree(dWeightList);
	cudaFree(dOutputList);
	cudaFree(dGradList);

	for (int i = 0; i < inputNum; i++) {
		if (batchNum > 1)
			nodeList[i]->localGrad += gradList[i] / batchNum;
		else
			nodeList[i]->localGrad = gradList[i];
	}

	delete gradList;
}

float Layer::getGrad(std::vector<float>& answerList, int batchNum) {
	int inputNum = nodeList.size();
	std::vector<float> inputList;
	std::vector<float> outputList;
	float *gradList = new float[inputNum];
	float *dInputList, *dOutputList, *dGradList, mse = 0;

	cudaMalloc(&dInputList, inputNum * sizeof(float));
	cudaMalloc(&dOutputList, inputNum * sizeof(float));
	cudaMalloc(&dGradList, inputNum * sizeof(float));

	for (int i = 0; i < inputNum; i++) {
		inputList.push_back(nodeList[i]->input);
		outputList.push_back(answerList[i] - nodeList[i]->output);
	}
	memcpy(gradList, outputList.data(), inputNum * sizeof(float));

	cudaMemcpy(dInputList, inputList.data(), inputNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dGradList, gradList, inputNum * sizeof(float), cudaMemcpyHostToDevice);

	nodeDelLog <<<1, inputNum >>> (dInputList, dGradList, sigmoidConst);
	cudaMemcpy(gradList, dGradList, inputNum * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < inputNum; i++) {
		if(batchNum > 1)
			nodeList[i]->localGrad += gradList[i]/batchNum;
		else
			nodeList[i]->localGrad = gradList[i];
	}
	cudaFree(dInputList);
	cudaFree(dOutputList);
	cudaFree(dGradList);

	delete gradList;
	for (int i = 0; i < inputNum; i++) {
		mse += outputList[i] * outputList[i];
	}
	mse /= inputNum;
	return mse;
}

void Layer::learnWeight(Layer& bLayer, float learningFactor) {
	std::vector<Node*> &bNodeList = bLayer.nodeList;
	int inputNum = bNodeList.size();
	int outputNum = nodeList.size();
	std::vector<float> inputList;
	std::vector<float> delList;
	float *weightList = new float[(inputNum + 1) * outputNum];
	float *dInputList, *dDelList, *dWeightList;
	dim3 threadGrid(inputNum + 1, outputNum);
	cudaMalloc(&dInputList, (inputNum + 1) * sizeof(float));
	cudaMalloc(&dDelList, outputNum * sizeof(float));
	cudaMalloc(&dWeightList, (inputNum + 1) * outputNum * sizeof(float));

	inputList.push_back(1);

	for (int i = 0; i < inputNum; i++) {
		inputList.push_back(bNodeList[i]->output);
	}

	inputNum++;

	cudaMemcpy(dInputList, inputList.data(), inputNum * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < outputNum; i++) {
		memcpy(weightList + i * inputNum, nodeList[i]->inputWeightList.data(), inputNum * sizeof(float));
		delList.push_back(nodeList[i]->localGrad);
	}

	cudaMemcpy(dWeightList, weightList, inputNum * outputNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dDelList, delList.data(), outputNum * sizeof(float), cudaMemcpyHostToDevice);

	nodeLearn <<<1, threadGrid >>> (dInputList, dDelList, dWeightList, learningFactor, inputNum);

	cudaMemcpy(weightList, dWeightList, inputNum * outputNum * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < outputNum; i++) {
		for (int j = 0; j < inputNum; j++) {
			nodeList[i]->inputWeightList[j] = weightList[i * inputNum + j];
		}
		nodeList[i]->localGrad = 0;
	}

	cudaFree(dInputList);
	cudaFree(dDelList);
	cudaFree(dWeightList);

	delete weightList;
}

void Layer::learnWeight(std::vector<float>& inputList, float learningFactor){
	int inputNum = inputList.size() + 1;
	int outputNum = nodeList.size();
	std::vector<float> delList;
	float *weightList = new float[inputNum * outputNum];
	float *dInputList, *dDelList, *dWeightList;
	dim3 threadGrid(inputNum, outputNum);
	cudaMalloc(&dInputList, inputNum * sizeof(float));
	cudaMalloc(&dDelList, outputNum * sizeof(float));
	cudaMalloc(&dWeightList, inputNum * outputNum * sizeof(float));

	inputList.insert(inputList.begin(), 1);

	cudaMemcpy(dInputList, inputList.data(), inputNum * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < outputNum; i++) {
		memcpy(weightList + i * inputNum, nodeList[i]->inputWeightList.data(), inputNum * sizeof(float));
		delList.push_back(nodeList[i]->localGrad);
	}

	cudaMemcpy(dWeightList, weightList, inputNum * outputNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dDelList, delList.data(), outputNum * sizeof(float), cudaMemcpyHostToDevice);

	nodeLearn <<<1, threadGrid >>> (dInputList, dDelList, dWeightList, learningFactor, inputNum);

	cudaMemcpy(weightList, dWeightList, inputNum * outputNum * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < outputNum; i++) {
		for (int j = 0; j < inputNum; j++) {
			nodeList[i]->inputWeightList[j] = weightList[i * inputNum + j];
		}
		nodeList[i]->localGrad = 0;
	}

	cudaFree(dInputList);
	cudaFree(dDelList);
	cudaFree(dWeightList);

	delete weightList;
	inputList.erase(inputList.begin());
}

__global__ void nodeCal(float* inputList, float* weightList, float* outputList){
	//int outputIdx = blockIdx.x * inputNum + threadIdx.x  ;
	int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
	float result = 0;
	extern __shared__ float results[];
	results[threadIdx.x] = inputList[threadIdx.x] * weightList[outputIdx];
	__syncthreads();
	//for (int i = 0; i < inputNum; i++) {
	for (int i = 0; i < blockDim.x; i++) {
		result += results[i];
	}
	outputList[blockIdx.x] = result;
}

__global__ void nodeLog(float* outputList, float sigmoidConst) {
	outputList[threadIdx.x] = tanh(sigmoidConst * outputList[threadIdx.x]);
}

__global__ void nodeGradCal(float* wList, float* outputList, float* gradList) {
	int weightIdx = blockIdx.x + threadIdx.x * gridDim.x;
	extern __shared__ float results[];
	float result = 0;
	results[threadIdx.x] = outputList[threadIdx.x] * wList[weightIdx];
	__syncthreads();
	for (int i = 0; i < blockDim.x; i++) {
		result += results[i];
	}
	gradList[blockIdx.x] = result;
}

__global__ void nodeDelLog(float* inputList, float* gradList, float sigmoidConst) {
	float temp;
	temp = cosh(sigmoidConst * inputList[threadIdx.x]);
	temp *= temp;
	temp = sigmoidConst / temp;
	gradList[threadIdx.x] *= temp;
}

__global__ void nodeLearn(float *inputList, float *delList, float *weightList, float learningFactor, int inputNum) {
	int weightIdx = threadIdx.x + threadIdx.y * inputNum;
	//int weightIdx = threadIdx.x + threadIdx.y * blockDim.x;
	weightList[weightIdx] += inputList[threadIdx.x] * delList[threadIdx.y] * learningFactor;
}
