#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
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
#include <sstream>

using std::cout;
using std::endl;

__global__ void relax(int N, int MAX_VAL, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_Di, int *d_out_P,  int *d_out_Pi) {
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);

    if (index < N - 1) {
        for (int j = d_in_I[index]; j < d_in_I[index + 1]; j++) {
            int u = d_in_V[index];
            //int v = d_in_V[d_in_E[j]];
            int w = d_in_W[j];
            int du = d_out_D[index];
            int dv = d_out_D[d_in_E[j]];
            int newDist = du + w;
            // Check if the distance is already set to max then just take the max since,
            // Cuda implementation gives this when a number is added to already max value of int.
            // E.g 2147483647 + 5 becomes -2147483644
            if (du == MAX_VAL){
                newDist = MAX_VAL;
            }
            //printf("Index = %d, w=%d, du =%d, dv=%d,  -- du + w = %d\n", index, w, du , dv, du + w);

            if (newDist < dv) {
                atomicExch(&d_out_Di[d_in_E[j]],newDist);
                atomicExch(&d_out_Pi[d_in_E[j]],u);
            }
        }
    }
}

__global__ void updateDistance(int N, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_Di, int *d_out_P,  int *d_out_Pi) {
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);
    if (index < N) {

        if (d_out_D[index] > d_out_Di[index]) {
            d_out_D[index] = d_out_Di[index];
        }
        if (d_out_P[index] != d_out_Pi[index]) {
            d_out_P[index] = d_out_Pi[index];
        }
        d_out_Di[index] = d_out_D[index];
        d_out_Pi[index] = d_out_P[index];
    }
}

__global__ void updateIndexOfEdges(int N, int *d_in_V, int *d_in_E, int l, int r) {
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);

    if(index ==1){
        printf("l=%d, r=%d\n",l,r);
    }
    // This does binary search on the V array to find the index of each node in the Edge array (E) and replace the same with index
    // Based on the iterative binary search from : https://www.geeksforgeeks.org/binary-search/
    if (index < N) {
        while (l <= r) {
            int m = l + (r - l) / 2;
            if(index == 1) {
                printf("m = %d, d_in_E[index] =%d\n",m, d_in_E[index]);
            }
            // Check if x is present at mid
            if (d_in_V[m] == d_in_E[index]) {
                d_in_E[index] = m;
                break;
            }
            // If x greater, ignore left half
            if (d_in_V[m] < d_in_E[index]) {
                l = m + 1;
            } else {        // If x is smaller, ignore right half
                r = m - 1;
            }
        }
    }
}

void loadVector(const char *filename, std::vector<int> &vec)
{
    std::ifstream input;
    input.open(filename);
    int num;
    while ((input >> num) && input.ignore()) {
        vec.push_back(num);
    }
    input.close();
}

void printVector(std::vector<int> &vec){
    for(int i=0; i<vec.size(); i++){
        cout<< vec[i] << " ";
    }
    cout<<endl;
}

void printCudaDevice(){
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
    }
}

void storeResult(const char *filename, std::vector<int> &V, int *D, int *P)
{
    std::ofstream output(filename);
    output << "Shortest Path : " << endl;
    for(int i = 0; i < V.size(); ++i)
    {
//        output << D[i] << ":" << P[i];
//        if (i != size-1){
//            output << ",";
//        }
        output << "from " << V[0] << " to " << V[i] << " = " << D[i] << " predecessor = " << P[i] << std::endl;
    }
    output.close();
}

int main (int argc, char **argv) {

    if (argc < 2 ){
        cout << "Input filename needs to be passed" << endl;
        cout << "example : ./par_bellman ../input/sample.gr <block-size> <debug>" << endl;
        cout << "block-size : optional param. Default 16" << endl;
        cout << "debug : optional param. 1 or 0. Prints additional log messages to console. Default 0" << endl;
        return -1;
    }
    std::string file=argv[1];
    int BLOCK_SIZE = 16;
    int debug;
    (argc == 3) ? BLOCK_SIZE=atoi(argv[2]) : BLOCK_SIZE=16;
    (argc == 4) ? debug=atoi(argv[3]) : debug=0;
    //input
    //std::vector<int> V = {1, 2, 3, 4, 5};
    //std::vector<int> I = {0, 2, 5, 6, 8, 10};
    //std::vector<int> E = {2, 4, 3, 4, 5, 2, 3, 5, 1, 3}; // This E stores destination vertex for each edge from V[I[i]].. V[I[i+1]]
    //std::vector<int> E = {1, 3, 2, 3, 4, 1, 2, 4, 0, 2}; // This E array stores index of destination vertex instead of actual vertex itself. So V[E[i]] is the vertex
    //std::vector<int> W = {6, 7, 5, 8, -4, -2, -3, 9, 2, 7};
    int MAX_VAL = std::numeric_limits<int>::max();

    std::vector<int> V, I, E, W;
    //Load data from files
    loadVector((file + "_V.csv").c_str(), V);
    loadVector((file + "_I.csv").c_str(), I);
    loadVector((file + "_E.csv").c_str(), E);
    loadVector((file + "_W.csv").c_str(), W);

    if(debug){
        cout << "V = "; printVector(V); cout << endl;
        cout << "I = "; printVector(I); cout << endl;
        cout << "E = "; printVector(E); cout << endl;
        cout << "W = "; printVector(W); cout << endl;
    }

    //output
    std::vector<int> D(V.size(), MAX_VAL); //Shortest path of V[i] from source
    std::vector<int> pred(V.size(), -1); // Predecessor vetex of V[i]

    //Set source vertex and predecessor
    D[0] = 0;
    pred[0] = 0;

    //int *in_V = V.data();
    //int *in_I = I.data();
    //int *in_E = E.data();
    //int *in_W = W.data();

    int N = I.size();
    int BLOCKS = 1;
    BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printCudaDevice();
    cout << "Blocks : " << BLOCKS << " Block size : " << BLOCK_SIZE << endl;

    int *d_in_V;
    int *d_in_I;
    int *d_in_E;
    int *d_in_W;
    int *d_out_D; // Final shortest distance
    int *d_out_Di; // Used in keep track of the distance during one single execution of the kernel
    int *d_out_P; // Final parent
    int *d_out_Pi; // Used in keep track of the parent during one single execution of the kernel

    //allocate memory
    cudaMalloc((void**) &d_in_V, V.size() *sizeof(int));
    cudaMalloc((void**) &d_in_I, I.size() *sizeof(int));
    cudaMalloc((void**) &d_in_E, E.size() *sizeof(int));
    cudaMalloc((void**) &d_in_W, W.size() *sizeof(int));

    cudaMalloc((void**) &d_out_D, D.size() *sizeof(int));
    cudaMalloc((void**) &d_out_Di, D.size() *sizeof(int));
    cudaMalloc((void**) &d_out_P, pred.size() *sizeof(int));
    cudaMalloc((void**) &d_out_Pi, pred.size() *sizeof(int));

    //copy to device memory
    cudaMemcpy(d_in_V, V.data(), V.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_I, I.data(), I.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_E, E.data(), E.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_W, W.data(), W.size() *sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_out_D, D.data(), D.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_P, pred.data(), pred.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_Di, D.data(), D.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_Pi, pred.data(), pred.size() *sizeof(int), cudaMemcpyHostToDevice);

    //Do binary search to find index of each element in E. This won't be necessary if the vertex starts from 0. But in the case of DIMACS vertex start from 1.
    //E.size() - because we need to replace each E[i] with its index of V[i]
    //0, V.size()-1 -for binary search of V array with each E[i] to find index
    updateIndexOfEdges<<<BLOCKS, BLOCK_SIZE>>>(E.size(), d_in_V, d_in_E, 0, V.size()-1);
    int *out_E_index = new int[E.size()];
    cudaMemcpy(out_E_index, d_in_E, E.size()*sizeof(int), cudaMemcpyDeviceToHost);

    const char *fname="../output/E_index.csv";
    std::ofstream output(fname);
    //std::ofstream output("../output/E_index.csv");
    for(int i = 0; i < E.size(); ++i){
        output << out_E_index[i];
        if(i != E.size()-2){
            output << ", ";
        }
    }
    output << endl;
    output.close();
    free(out_E_index);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Bellman ford
    for (int round = 1; round < V.size(); round++) {
        if(debug){
            cout<< "***** round = " << round << " ******* " << endl;
        }
        relax<<<BLOCKS, BLOCK_SIZE>>>(N, MAX_VAL, d_in_V, d_in_I, d_in_E, d_in_W, d_out_D, d_out_Di, d_out_P, d_out_Pi);
        updateDistance<<<BLOCKS, BLOCK_SIZE>>>(N, d_in_V, d_in_I, d_in_E, d_in_W, d_out_D, d_out_Di, d_out_P, d_out_Pi);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    int *out_path = new int[V.size()];
    int *out_pred = new int[V.size()];

    cudaMemcpy(out_path, d_out_D, D.size()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_pred, d_out_P, pred.size()*sizeof(int), cudaMemcpyDeviceToHost);

    if(debug) {
        cout << "Shortest Path : " << endl;
        for (int i = 0; i < D.size(); i++) {
            cout << "from " << V[0] << " to " << V[i] << " = " << out_path[i] << " predecessor = " << out_pred[i]
                 << std::endl;
        }
    }

    // Create output file name by parsing the input filename and extracting the last string :
    std::string delimiter = "/";
    size_t pos = 0;
    std::string token;
    while ((pos = file.find(delimiter)) != std::string::npos) {
        token = file.substr(0, pos);
        //std::cout << token << std::endl;
        file.erase(0, pos + delimiter.length());
    }
    //cout << file << endl;
    storeResult(("../output/" + file + "_SP.csv").c_str(),V, out_path, out_pred);

   cout << "average time elapsed : " << elapsedTime << endl;

    free(out_pred);
    free(out_path);
    cudaFree(d_in_V);
    cudaFree(d_in_I);
    cudaFree(d_in_E);
    cudaFree(d_in_W);
    cudaFree(d_out_D);
    cudaFree(d_out_P);
    cudaFree(d_out_Di);
    cudaFree(d_out_Pi);
}
