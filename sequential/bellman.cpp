#include "../main.h"

void updateIndexOfEdges(std::vector<int> &V, std::vector<int> &E, int l, int r){

    for (int index = 0; index < E.size(); index++) {
        //cout << "Updating index of  E[index] " <<  E[index] << endl;
        l=0; r=V.size()-1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            // Check if x is present at mid
            if (V[m] == E[index]) {
                E[index] = m;
                break;
            }
            // If x greater, ignore left half
            if (V[m] < E[index]) {
                l = m + 1;
            } else {        // If x is smaller, ignore right half
                r = m - 1;
            }
        }
        //cout << "index of  E[index] " <<  E[index] << endl;
    }
}
void runBellmanFordSequential(std::string file, int debug){

    cout << "Running BellmanFord Sequential for : " << file << endl;
    std::vector<int> V, I, E, W;
    //Load data from files
    loadVector((file + "_V.csv").c_str(), V);
    loadVector((file + "_I.csv").c_str(), I);
    loadVector((file + "_E.csv").c_str(), E);
    loadVector((file + "_W.csv").c_str(), W);

    std::vector<int> D(V.size(), std::numeric_limits<int>::max());
    std::vector<int> pred(V.size(), -1);

    updateIndexOfEdges(V, E, 0, V.size()-1);

    D[0] = 0;
    pred[0] = 0;

    // Bellman ford
    for (int round = 1; round < V.size(); round++) {
        if(debug){
            cout<< "***** round = " << round << " ******* " << endl;
        }
        for (int i = 0; i < I.size()-1 ; i++) {
            for (int j = I[i]; j < I[i + 1]; j++) {
                int u = V[i];
                int v = V[E[j]];
                int w = W[j];
                int du = D[i];
                int dv = D[E[j]];
                if (du + w < dv) {
                    //cout<< "Relaxing edge (" << u << ", " << v << ") current dist =" << dv << ", new dist =" << du + w << endl;
                    D[E[j]] = du + w;
                    pred[E[j]] = u;
                }
            }
        }
    }

    storeResult(("../output/" + makeOutputFileName(file) + "_SP_Sequential.csv").c_str(), V,D.data(),pred.data());
    cout << "Results written to " << ("../output/" + makeOutputFileName(file) + "_SP_Sequential.csv").c_str() << endl;

    if (debug) {
        cout << "Shortest Path : " << endl;
        for (int i = 0; i < V.size(); i++) {
            cout << "from " << V[0] << " to " << V[i] << " = " << D[i] << " predecessor = " << pred[i] << std::endl;
        }
    }
}