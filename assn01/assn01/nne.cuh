#pragma once
#include <vector>

class Node {
public:
	Node();//first input: constant/offset = 0
	Node(int inputNum);//inputNum: nuber of input (first input: constant/offset = 0)
	~Node();
	std::vector<float> inputWeightList;
	float output;
};

class Layer {
public:
	std::vector<Node> nodeList;
	void backPropa(Layer& layer);
};
