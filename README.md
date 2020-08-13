# Bellman Ford Single Source Shortest Path Algorithm on GPU using CUDA
Term Project, Parallel Algorithms, Summer 2020, University of Texas, Austin

## Compiling the project

1. Connect to `maverick2.tacc.utexas.edu`
2. Upload the files
3. `mkdir build/`
4. `cd build`
5. `cmake ../.`
6. `make` - Builds executable `bellman`
7. `$ ./bellman seq ../input/simple.gr` - Runs sequential version
8. `$./bellman cuda ../input/USA-road-d.NY.gr 196 1024 0` - Runs cuda implementation version 1
9. `$./bellman cuda-stride ../input/USA-road-d.NY.gr 196 1024 0` - Runs cuda implementation version 2
10. `$./bellman cuda-v3 ../input/USA-road-d.NY.gr 196 1024 0` - Runs cuda implementation version 3

## Implementation Details
* Folder `sequential` contains base sequential implementation of BellmanFord. 
* Folder `gpu` contains cuda implementation of BellmanFord
* Folder `utilities` contains common utility functions used by both the implementations 
* Folder `parser`contains the parser utility to covert DIMACS graph files (`http://users.diag.uniroma1.it/challenge9/format.shtml#ss`) into CSR format. Run parser using `$ ./parser ./input/USA-road-d.NY.gr` command
* Folder `input` contains graph files in CSR format
* `main.cpp` - Runs both sequential and cuda versions


###### Sample Graph:
<div>
<img src="https://user-images.githubusercontent.com/48846576/89080545-cb4dba00-d34e-11ea-8dbd-6e7f4b897bb5.png" height="250" width="300"/>
</div>

###### CSR Representation Graph:
<div>
<img src="https://user-images.githubusercontent.com/48846576/89236974-ac9e2c00-d5b7-11ea-9996-dca858eb0535.jpg" height="250" width="400"/>
</div>

- `V` : array of vertices of size `|V|`
- `I` : array of starting index of the adjacency list of edges in `E` array. Size `|V+1|`. The last element stores `|E|`
- `E` : array of edges `|E|`
- `W` : array of weights `|W|`
 
###### Shortest Path : `1 -> 4 -> 3 -> 2 -> 5`
- from 1 to 1 = 0, predecessor = 0
- from 1 to 4 = 7, predecessor = 1
- from 1 to 3 = 4, predecessor = 4
- from 1 to 2 = 2, predecessor = 3
- from 1 to 5 = -2, predecessor = 

## Input Data

- `input` folder contains random and USA road networks graphs from DIMACS in CSR format
- Each array in the CSR format is stored in separate CSV files which are read by the CUDA program
- `parser/parser.go` converts the DIMACS files into CSR format and stores in individual csv files 
- Example: `USA-road-d.NY.gr` file from `http://users.diag.uniroma1.it/challenge9/download.shtml` has been transformed into the below ones
    - `USA-road-d.NY.gr_V.csv` - Contains V array (as depicted in figure above)
    - `USA-road-d.NY.gr_I.csv` - Contains I array
    - `USA-road-d.NY.gr_E.csv` - Contains E array
    - `USA-road-d.NY.gr_W.csv` - Contains W array
    - `USA-road-d.NY.gr_FROM.csv` & `USA-road-d.NY.gr_TO.csv`- Contains all edges of the graph where source is in FROM and destination vertex is in TO (Will be useful for version 2 stated below)

## Bellman Ford GPU Implementation
Implement and study the performance in three different flavors of the algorithm
- [x] Version 1 - One thread to each vertex (to relax all outgoing edges of each vertex) - Number of Blocks is determined based on input nodes. More threads & each thread doing less work  
- [x] Version 2 - Introduce stride inside kernel. Fixed number of blocks. Less threads and each thread doing more work
- [x] Version 3 - Do relaxation for each V[i] only if Flag[i] is set to true. i.e. if V[i] has shorter distance than the previous iteration.

## Results

![Performance Analysis](https://user-images.githubusercontent.com/48846576/90195833-1ffe2580-dd90-11ea-8dfd-54e0000483b8.png)    
![Graph_results](https://user-images.githubusercontent.com/48846576/90195829-1e346200-dd90-11ea-9205-437722d3789b.png)
    

## References
- Shortest Paths Algorithms: Theory And ExperimentalEvaluation. Boris Cherkassky, Andrew V. Goldberg and Tomasz Radzik
- New Approach of Bellman Ford Algorithm on GPU using Compute Unified Design Architecture (CUDA) - Agarwal, Pankhari, Dutta, Maitreyee 
- Accelerating large graph algorithms on the GPU using CUDA - Pawan Harish and P. J. Narayanan
- https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
- https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/