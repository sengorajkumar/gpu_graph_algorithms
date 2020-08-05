//
// Created by rajkumar sengottuvel on 8/4/20.
//

#include "utilities.h"

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

std::string makeOutputFileName(std::string inputFile){
    std::string delimiter = "/";
    size_t pos = 0;
    std::string token;
    while ((pos = inputFile.find(delimiter)) != std::string::npos) {
        token = inputFile.substr(0, pos);
        inputFile.erase(0, pos + delimiter.length());
    }
    return inputFile;
}

void storeResult(const char *filename, std::vector<int> &V, int *D, int *P)
{
    std::ofstream output(filename);
    output << "Shortest Path : " << endl;
    for(int i = 0; i < V.size(); ++i)
    {
        output << "from " << V[0] << " to " << V[i] << " = " << D[i] << " predecessor = " << P[i] << std::endl;
    }
    output.close();
}

bool isValidFile (const std::string& filename) {
    ifstream file(filename.c_str());
    return file.good();
}