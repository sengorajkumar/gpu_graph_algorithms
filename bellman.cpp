#include <iostream>
#include <vector>
#include <limits>

using std::cout;
using std::endl;

void BellmanFord(){

    std::vector<int> V = {1, 2, 3, 4, 5};
    std::vector<int> I = {0, 2, 5, 6, 8, 10};
    //std::vector<int> E = {2, 4, 3, 4, 5, 2, 3, 5, 1, 3}; // This E stores destination vertex for each edge from V[I[i]].. V[I[i+1]]
    std::vector<int> E = {1, 3, 2, 3, 4, 1, 2, 4, 0, 2}; // This E array stores index of destination vertex instead of actual vertex itself. So V[E[i]] is the vertex
    std::vector<int> W = {6, 7, 5, 8, -4, -2, -3, 9, 2, 7};

    std::vector<int> D(V.size(), std::numeric_limits<int>::max());
    std::vector<int> pred(V.size(), -1);

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
                    cout<< "Relaxing edge (" << u << ", " << v << ") current dist =" << dv << ", new dist =" << du + w << endl;
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

int main(){
    cout<< "Running sequential bellman ford" << endl;
    BellmanFord();
}