#include "../gpu/main.h"

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
void BellmanFord(std::string file){

    //std::vector<int> V = {1, 2, 3, 4, 5};
    //std::vector<int> I = {0, 2, 5, 6, 8, 10};
    //std::vector<int> E = {2, 4, 3, 4, 5, 2, 3, 5, 1, 3}; // This E stores destination vertex for each edge from V[I[i]].. V[I[i+1]]
    //std::vector<int> E = {1, 3, 2, 3, 4, 1, 2, 4, 0, 2}; // This E array stores index of destination vertex instead of actual vertex itself. So V[E[i]] is the vertex
    //std::vector<int> W = {6, 7, 5, 8, -4, -2, -3, 9, 2, 7};

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
        cout<< "***** round = " << round << " ******* " << endl;
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

    cout << "Shortest Path : " << endl;
    for (int i = 0; i < V.size(); i++) {
        cout << "from " << V[0] << " to " << V[i] << " = " << D[i] << " predecessor = " << pred[i] << std::endl;
    }

}