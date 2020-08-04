#include "main.h"

int main(int argc, char **argv) {
    if (argc < 2 ){
        cout << "Input filename needs to be passed" << endl;
        return -1;
    }
    std::string file=argv[1];
    cout<< "Running sequential bellman ford" << endl;
    // Reference https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    BellmanFord(file);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Elapsed time : " << duration.count() << "milli seconds " << endl;
}