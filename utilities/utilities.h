#ifndef GPU_GRAPH_ALGORITHMS_UTILITIES_H
#define GPU_GRAPH_ALGORITHMS_UTILITIES_H

#include<vector>
#include <fstream>
#include <iostream>
using namespace std;
using std::cout;
using std::endl;

void loadVector(const char *filename, std::vector<int> &vec);
void printVector(std::vector<int> &vec);
void storeResult(const char *filename, std::vector<int> &V, int *D, int *P);
bool isValidFile (const std::string& filename);
std::string makeOutputFileName(std::string inputFile);
#endif //GPU_GRAPH_ALGORITHMS_UTILITIES_H
