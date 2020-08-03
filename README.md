# Single Source Shortest Path Algorithms on GPU using CUDA
Term Project, Parallel Algorithms, Summer 2020, University of Texas, Austin

## Compiling the project

1. Connect to `maverick2.tacc.utexas.edu`
2. Upload the files
3. `mkdir build/`
4. `cd build`
5. `cmake ../.`
6. `make seq_bellman` - Sequential Bellman Ford for the sample graph below
7. `make par_bellman` - Parallel Bellman Ford on GPU 

## Bellman Ford
* Sequential implementation of Bellman ford can be found in `bellman.cpp`. 
* GPU implementation can be found in `par_bellman.cu`
* Both versions use the below sample graph in CSR format.

###### Sample Graph:
![Sample_Graph_For_Bellman](https://user-images.githubusercontent.com/48846576/89080545-cb4dba00-d34e-11ea-8dbd-6e7f4b897bb5.png)

###### CSR Representation:
![CSR_Format](https://user-images.githubusercontent.com/48846576/89236974-ac9e2c00-d5b7-11ea-9996-dca858eb0535.jpg)

- `V` : array of vertices of size `|V|`
- `I` : array of starting index of the adjacency list of edges in `E` array. Size `|V+1|`. The last element stores `|E|`
- `E` : array of edges `|E|`
- `W` : array of weights `|W|`
 
###### Shortest Path : `1 -> 4 -> 3 -> 2 -> 5`
- from 1 to 1 = 0, predecessor = 0
- from 1 to 4 = 7, predecessor = 1
- from 1 to 3 = 4, predecessor = 4
- from 1 to 2 = 2, predecessor = 3
- from 1 to 5 = -2, predecessor = 2