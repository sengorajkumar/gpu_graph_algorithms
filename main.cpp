#include "main.h"

int main(int argc, char **argv) {
    if (argc < 2 ){
        cout << "Usage : ./bellman MODE FILE BLOCK_SIZE DEBUG" << endl;
        cout << "MODE - seq / cuda \n"
                "FILE - Input file \n"
                "BLOCK_SIZE - Number of threads per block for cuda \n"
                "DEBUG - 1 or 0 to enable/disable extended debug messages on console\n"
                "Program expects these CSV files based on FILE thats passed in the argument\n"
                "    FILE_V.csv\n"
                "    FILE_I.csv\n"
                "    FILE_E.csv\n"
                "    FILE_W.csv"
                << endl;
        return -1;
    }
    std::string mode = argv[1];
    std::string file;
    int debug;
    int BLOCK_SIZE;
    int BLOCKS;
    if(argv[2] != NULL){
        file = argv[2];
        //Check if all CSR files are present
        if(!isValidFile(file + "_V.csv") ||
           !isValidFile(file + "_I.csv") ||
           !isValidFile(file + "_E.csv") ||
           !isValidFile(file + "_W.csv")){
            cout << "One or more CSR files missing" << endl;
            return -1;
        }

    }
    if(argv[3] != NULL){
        BLOCK_SIZE = atoi(argv[3]);
    }
    
    if(argv[4] != NULL){
        BLOCKS = atoi(argv[4]);
    }

    if(argv[5] != NULL){
        debug = atoi(argv[5]);
    }

    (BLOCK_SIZE == 0) ? 512 : BLOCK_SIZE; // Set default to 512 threads
    (debug == 0) ? 0 : 1;

    if(mode == "seq") {
        // Reference https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
        auto start = high_resolution_clock::now();
        runBellmanFordSequential(file, debug);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        //cout << "Elapsed time : " << duration.count() << " milli seconds " << endl;
        //cout << duration.count() << endl;
    }
    if(mode == "cuda"){
        runBellmanFordOnGPU(file.c_str(), BLOCK_SIZE, BLOCKS, debug);
    }
}