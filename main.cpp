#include "main.h"

int main(int argc, char **argv) {
    if (argc < 2 ){
        cout << "Usage : ./bellman MODE FILE BLOCK_SIZE DEBUG" << endl;
        cout << "MODE - seq / cuda \n"
                "FILE - Input file \n"
                "BLOCK_SIZE - Number of threads per block for cuda \n"
                "DEBUG - 1 or 0 to enable/disable extended debug messages on console" << endl;
        return -1;
    }
    std::string mode = argv[1];
    std::string file;
    int debug;
    int BLOCK_SIZE;
    if(argc == 3){
        file = argv[2];
    }
    if(argc == 4){
        BLOCK_SIZE = atoi(argv[3]);
    }
    if(argc == 5){
        debug = atoi(argv[3]);
    }

    (BLOCK_SIZE == 0) ? 512 : BLOCK_SIZE; // Set default to 512 threads
    (debug == 0) ? 0 : 1;

    if(mode == "seq") {
        // Reference https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
        auto start = high_resolution_clock::now();
        runBellmanFordSequential(file);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "Elapsed time : " << duration.count() << " milli seconds " << endl;
    }
    if(mode == "cuda"){
        runBellmanFordOnGPU(file.c_str(), BLOCK_SIZE,debug);
    }
}