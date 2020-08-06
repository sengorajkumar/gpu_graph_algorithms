#include "kernels.cuh"

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

    // This does binary search on the V array to find the index of each node in the Edge array (E) and replace the same with index
    // Based on the iterative binary search from : https://www.geeksforgeeks.org/binary-search/
    if (index < N) {
        while (l <= r) {
            int m = l + (r - l) / 2;
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

__global__ void initializeArray(const int N, int *p, const int val, bool sourceDifferent, const int source, const int sourceVal){
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    p[index] = val;
    if(sourceDifferent){
        if(index == source) {
            p[index] = sourceVal;
        }
    }
}