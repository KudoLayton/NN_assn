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
	Layer(int nodeListLength, int inputWeightLength, float sigmoidConst = 0.01);
	Layer(std::vector<float>& inputWeightList, int nodeListLength, int inputWeightLength, float sigmoidConst = 0.01);
	~Layer();
	std::vector<Node*> nodeList;
	float sigmoidConst;
	void forwardCal(Layer& bLayer);
	void forwardCal(std::vector<float>& inputList);
	void getGrad(Layer& fLayer);
	float getGrad(std::vector<float>& answerList); // output: MSE
	void learnWeight(Layer& bLayer, float learningFactor);
	void learnWeight(std::vector<float>& inputList, float learningFactor);
};
