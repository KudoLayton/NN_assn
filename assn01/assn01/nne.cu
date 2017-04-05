#include "nne.cuh"
#include <cstdlib>

Node::Node() : output(0){
	inputWeightList.push_back(0);
}

Node::Node(int inputNum) : output(0) {
	inputWeightList.push_back(0);
	for (int i = 0; i < inputNum; i++) {
		inputWeightList.push_back((float)rand() / RAND_MAX);
	}
}

Node::Node(std::vector<float>& inputWeightList, int nodeIndex, int inputWeightLength) : output(0) {
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
	}
}

Layer::Layer(std::vector<float>& inputWeightList, int nodeListLength, int inputWeightLength) {
	Node* newNode;
	for (int i = 0; i < nodeListLength; i++) {
		newNode = new Node(inputWeightList, i, inputWeightLength);
	}
}

Layer::~Layer() {
	int length = nodeList.size();
	for (int i = 0; i < length; i++) {
		delete nodeList[i];
	}
}

void Layer::backPropa(Layer& layer) {
	
}

__global__ void nodeCal(float* inputList, float* weightList, float* outputList){
	
}
