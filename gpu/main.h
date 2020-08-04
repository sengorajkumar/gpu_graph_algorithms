//
// Created by rajkumar sengottuvel on 8/4/20.
//

#ifndef GPU_GRAPH_ALGORITHMS_MAIN_H
#define GPU_GRAPH_ALGORITHMS_MAIN_H
#include <iostream>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <stdio.h>
#include <string>
#include <string.h>
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;
using std::cout;
using std::endl;

void BellmanFord(std::string file);
void loadVector(const char *filename, std::vector<int> &vec);
void updateIndexOfEdges(std::vector<int> &V, std::vector<int> &E, int l, int r);

#endif //GPU_GRAPH_ALGORITHMS_MAIN_H
