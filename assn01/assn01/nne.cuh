#pragma once
#include <vector>
#include "cuda_runtime.h"

__global__ void nodeCal(float* inputList, float* weightList, float* outputList);

class Node {
public:
	Node();//first input: constant/offset = 0
	Node(int inputNum);//inputNum: nuber of input (first input: constant/offset = 0)
	Node(std::vector<float>& inputWeightList, int nodeIndex, int inputWeightLength);
	~Node();
	std::vector<float> inputWeightList;
	float output;
};

class Layer {
public:
	Layer();
	Layer(int nodeListLength, int inputWeightLength);
	Layer(std::vector<float>& inputWeightList, int nodeListLength, int inputWeightLength);
	~Layer();
	std::vector<Node*> nodeList;
	void backPropa(Layer& layer);
};
