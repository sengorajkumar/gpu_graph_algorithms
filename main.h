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
#include "utilities/utilities.h"

using namespace std;
using namespace std::chrono;
using std::cout;
using std::endl;

void runBellmanFordSequential(std::string file, int debug);
int runBellmanFordOnGPU(const char *file, int blockSize, int debug);
int runBellmanFordOnGPUWithGridStride(const char *file, int blocks, int blockSize, int debug);

#endif //GPU_GRAPH_ALGORITHMS_MAIN_H
