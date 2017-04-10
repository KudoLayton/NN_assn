#include "cuda_runtime.h"
#include "nne.cuh"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <bitset>
#include <fstream>
#include <string>
#include <stdlib.h>

using namespace std;
char inputString[1000];
union{
	float ff1;
	int bset1;
}d1;
union{
	float ff2;
	unsigned bset2;
}d2;
int ii;

int main()
{
	Layer inputLayer(64, 64);
	Layer hiddenLayer(16, 64);
	Layer outputLayer(2, 16);
	std::vector<float> inputList;
	std::vector<float> outputList;
	float EMS;
	srand((unsigned int)time(NULL));
	cudaSetDevice(0);

	for (int i = 0; i < 20; i++) {
		ifstream inFile("two_moon.txt");
		while (!inFile.eof()){
			inFile.getline(inputString, 100);
			d1.ff1 = atof(inputString);
			bitset<32> b1(d1.bset1);
			for (int i = 0; i < 32; i++){
				inputList.push_back(b1[i]);
			}
			inFile.getline(inputString, 100);
			d2.ff2 = atof(inputString);
			bitset<32> b2(d2.bset2);
			for (int i = 0; i < 32; i++){
					inputList.push_back(b2[i]);
			}
			inFile.getline(inputString, 100);
			ii = atoi(inputString);
			outputList.push_back(ii);

			for (int i = 0; i < 64; i++){
				//cout<<inputList.at(i);
			}//cout<<d1.ff1<<endl;
			//cout<<d2.ff2<<endl;

			if (ii == 0){
				outputList.push_back(1);
				outputList.push_back(0);
			}else{
				outputList.push_back(0);
				outputList.push_back(1);
			}

			inputLayer.forwardCal(inputList);
			hiddenLayer.forwardCal(inputLayer);
			outputLayer.forwardCal(hiddenLayer);
			EMS = outputLayer.getGrad(outputList);
			std::cout << EMS << std::endl;
			hiddenLayer.getGrad(outputLayer);
			inputLayer.getGrad(hiddenLayer);
			inputLayer.learnWeight(inputList, 0.7);
			hiddenLayer.learnWeight(inputLayer, 0.7);
			outputLayer.learnWeight(hiddenLayer, 0.7);

			inputList.clear();
			outputList.clear();
		}
		inFile.close();
	}

	cudaDeviceReset();
	return 0;
}
