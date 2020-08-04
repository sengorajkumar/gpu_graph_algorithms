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
- Example: `USA-road-d.NY.gr` file from `http://users.diag.uniroma1.it/challenge9/download.shtml` has been transformed into the below ones
    - `USA-road-d.NY.gr_V.csv` - Contains V array (as depicted in figure above)
    - `USA-road-d.NY.gr_I.csv` - Contains I array
    - `USA-road-d.NY.gr_E.csv` - Contains E array
    - `USA-road-d.NY.gr_W.csv` - Contains V array
    - `USA-road-d.NY.gr_FROM.csv` & `USA-road-d.NY.gr_TO.csv`- Contains all edges of the graph where source is in FROM and destination vertex is in TO
    
| File | Nodes | Edges |
| :---         |     :---:      |        :---: |
| rand_1000   | 1000     | 5000    |
| USA-road-d.NY     |  264,346       |  733,846     |
| USA-road-d.COL     |  435,666       |  1,057,066     |
| add few more random and road networks     |  -       |  -     |
    
## Bellman Ford GPU Implementation
Implement and study the performance in three different flavors of the algorithm
- [x] Version 1 - One thread to each vertex (to relax all outgoing edges of each vertex)
- [ ] Version 2 - One thread for each edge. Slight variation of version 1
- [ ] Version 3 - Have a boolean flag to check whether to continue till V-1 rounds or terminate early.
