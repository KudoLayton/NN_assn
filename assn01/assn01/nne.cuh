#pragma once
#include <vector>
#include <list>
#include "cuda_runtime.h"

class Node {
public:
	Node();//first input: constant/offset = 0
	Node(int inputNum);//inputNum: nuber of input (first input: constant/offset = 0)
	Node(std::vector<float>& inputWeightList, int nodeIndex, int inputWeightLength);
	float getMiniBatchGrad(float newGrad, int batchNum);
	~Node();
	std::vector<float> inputWeightList;
	float output;
	float input;
	float localGrad;
private:
	std::list<float> localGradList;
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
	void getGrad(Layer& fLayer, int batchNum = 1);
	float getGrad(std::vector<float>& answerList, int batchNum = 1); // output: MSE
	void learnWeight(Layer& bLayer, float learningFactor);
	void learnWeight(std::vector<float>& inputList, float learningFactor);
};
