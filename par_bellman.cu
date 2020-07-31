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

using std::cout;
using std::endl;

__global__ void relax(int N, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_pred) {
    unsigned int tid = threadIdx.x;
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);

    if (index < N - 1) {
        for (int j = d_in_I[index]; j < d_in_I[index + 1]; j++) {
            int u = d_in_V[index];
            //int v = d_in_V[d_in_E[j]];
            int w = d_in_W[j];
            int du = d_out_D[index];
            int dv = d_out_D[d_in_E[j]];
            if (du + w < dv) {
                atomicExch(&d_out_D[d_in_E[j]], du + w);
                atomicExch(&d_out_pred[d_in_E[j]], u);
            }
        }
    }
}

int main (int argc, char **argv) {

    //input
    std::vector<int> V = {1, 2, 3, 4, 5};
    std::vector<int> I = {0, 2, 5, 6, 8, 10};
    //std::vector<int> E = {2, 4, 3, 4, 5, 2, 3, 5, 1, 3}; // This E stores destination vertex for each edge from V[I[i]].. V[I[i+1]]
    std::vector<int> E = {1, 3, 2, 3, 4, 1, 2, 4, 0, 2}; // This E array stores index of destination vertex instead of actual vertex itself. So V[E[i]] is the vertex
    std::vector<int> W = {6, 7, 5, 8, -4, -2, -3, 9, 2, 7};

    //output
    std::vector<int> D(V.size(), std::numeric_limits<int>::max()); //Shortest path of V[i] from source
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
    int BLOCK_SIZE = 16;
    BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

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

    cout << "Blocks : " << BLOCKS << " Block size : " << BLOCK_SIZE << endl;

    int *d_in_V;
    int *d_in_I;
    int *d_in_E;
    int *d_in_W;
    int *d_out_D;
    int *d_out_pred;

    //allocate memory
    cudaMalloc((void**) &d_in_V, V.size() *sizeof(int));
    cudaMalloc((void**) &d_in_I, I.size() *sizeof(int));
    cudaMalloc((void**) &d_in_E, E.size() *sizeof(int));
    cudaMalloc((void**) &d_in_W, W.size() *sizeof(int));

    cudaMalloc((void**) &d_out_D, V.size() *sizeof(int));
    cudaMalloc((void**) &d_out_pred, V.size() *sizeof(int));

    //copy to device memory
    cudaMemcpy(d_in_V, V.data(), V.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_I, I.data(), I.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_E, E.data(), E.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_W, W.data(), W.size() *sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_out_D, D.data(), D.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_pred, pred.data(), pred.size() *sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Bellman ford
    for (int round = 1; round < V.size(); round++) {
        cout<< "***** round = " << round << " ******* " << endl;
        relax<<<BLOCKS, BLOCK_SIZE>>>(N, d_in_V, d_in_I, d_in_E, d_in_W, d_out_D, d_out_pred);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    int *out_path = new int[V.size()];
    int *out_pred = new int[V.size()];

    cudaMemcpy(out_path, d_out_D, D.size()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_pred, d_out_pred, pred.size()*sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Shortest Path : " << endl;
    for (int i = 0; i < D.size(); i++) {
        cout << "from " << V[0] << " to " << V[i] << " = " << out_path[i] << " predecessor = " << out_pred[i] << std::endl;
    }

    cout << "average time elapsed : " << elapsedTime << endl;

    free(out_pred);
    free(out_path);
    cudaFree(d_in_V);
    cudaFree(d_in_I);
    cudaFree(d_in_E);
    cudaFree(d_in_W);
    cudaFree(d_out_D);
    cudaFree(d_out_pred);
}
