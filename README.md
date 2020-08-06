# Bellman Ford Single Source Shortest Path Algorithm on GPU using CUDA
Term Project, Parallel Algorithms, Summer 2020, University of Texas, Austin

## Compiling the project

1. Connect to `maverick2.tacc.utexas.edu`
2. Upload the files
3. `mkdir build/`
4. `cd build`
5. `cmake ../.`
6. `make` - Builds executable `bellman`
7. `$ ./bellman seq ../input/simple.gr` - Runs sequential version of bellman ford
8. `$ ./bellman cuda ../input/simple.gr 1024` - Runs cuda version of bellman ford

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
    
| File           |	nodes |	edges|	Time (Milli seconds)| Time(min)	|	TACC command |
| :---          |   :---    |:---   |:---   |:---   |:---   |
|USA-road-d.NY      |	264,346	|733,846	|17293.9|	0.29	|`sbatch run_bellman_cuda.sh ../input/USA-road-d.NY.gr 1024`|
|USA-road-d.COL     |	435,666	|1,057,066|	39291.9|	0.65|	`sbatch run_bellman_cuda.sh ../input/USA-road-d.COL.gr 1024`|
|USA-road-d.FLA     |	1,070,376|	2,712,798|	229136|	3.82	|`sbatch run_bellman_cuda.sh ../input/USA-road-d.FLA.gr 1024`|
|USA-road-d.CAL     |	1,890,815	|4,657,742|	764928|	12.75|	`sbatch run_bellman_cuda.sh ../input/USA-road-d.CAL.gr 1024`|
|USA-road-d.E	    |3,598,623	|8,778,114|	2.88E+06|	47.94	|`sbatch run_bellman_cuda.sh ../input/USA-road-d.E.gr 1024`  |  

## Bellman Ford GPU Implementation
Implement and study the performance in three different flavors of the algorithm
- [x] Version 1 - One thread to each vertex (to relax all outgoing edges of each vertex)
- [ ] Version 2 - One thread for each edge. Slight variation of version 1
- [ ] Version 3 - Have a boolean flag to check whether to continue till V-1 rounds or terminate early.

## References
- Agarwal, Pankhari, Dutta, Maitreyee - New Approach of Bellman Ford Algorithm on GPU using Compute Unified Design Architecture (CUDA)
- 