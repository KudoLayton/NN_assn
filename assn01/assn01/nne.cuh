#pragma once
#include <vector>
#include "cuda_runtime.h"

class Node {
public:
	Node();//first input: constant/offset = 0
	Node(int inputNum);//inputNum: nuber of input (first input: constant/offset = 0)
	Node(std::vector<float>& inputWeightList, int nodeIndex, int inputWeightLength);
	~Node();
	std::vector<float> inputWeightList;
	float output;
	float input;
	float localGrad;
};

class Layer {
public:
	Layer();
	Layer(int nodeListLength, int inputWeightLength);
	Layer(std::vector<float>& inputWeightList, int nodeListLength, int inputWeightLength);
	~Layer();
	std::vector<Node*> nodeList;
	void forwardCal(Layer& bLayer);
	void forwardCal(std::vector<float>& inputList);
	void getGrad(Layer& fLayer);
	float backPropa(std::vector<float>& answerList, float learningFactor);
	void backPropa(Layer& fLayer, float learningFactor);
};
