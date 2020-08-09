package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

func LoadGraphV1(filename string) Graph {

	graph := Graph{edges: make(map[string][]Edge), nodes: HashSet{elements: make(map[string]bool)}}
	file, _ := os.Open(filename)
	defer file.Close()
	fscanner := bufio.NewScanner(file)
	for fscanner.Scan() {
		line := fscanner.Text()
		edgeData := strings.Fields(line)
		if edgeData[0] == "a" {
			edge := Edge{u: "", v: "", w: 0}
			edge.u = edgeData[1]
			edge.v = edgeData[2]
			weight, _ := strconv.Atoi(edgeData[3])
			edge.w = weight
			graph.AddEdgeAndNodes(edgeData[1], edge)
		}
	}
	return graph
}

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		fmt.Println("Input filename needs to be passed as an argument")
		return
	}
	fullname := args[0]
	s := strings.Split(fullname, "/")
	filename := s[len(s)-1]
	g := LoadGraphV1(fullname)
	fmt.Println("File loaded ", fullname)

	vertices := g.nodes.elements

	V := make([]int, 0)
	for key, _ := range vertices {
		node, _ := strconv.Atoi(key)
		V = append(V, node)
	}
	sort.Ints(V) // Sort the array for binary search later
	fmt.Println("Sorting of Vertices done")

	I := make([]int, 0)
	E := make([]int, 0)
	W := make([]int, 0)
	FROM := make([]int, 0)
	TO := make([]int, 0)
	counter := 0
	fmt.Println("Creating I E W arrays")
	for _, node := range V {
		I = append(I, counter)
		key := strconv.Itoa(node)
		edges := g.edges[key]
		for _, edge := range edges {
			u, _ := strconv.Atoi(edge.u)
			v, _ := strconv.Atoi(edge.v)
			w := edge.w
			E = append(E, v)
			W = append(W, w)
			FROM = append(FROM, u)
			TO = append(TO, v)
			counter++
		}
	}
	I = append(I, g.NumEdges())
	fmt.Println("num edges : ", counter)
	fmt.Println("Writing to files")
	WriteData("./output/"+filename+"_V.csv", V)
	WriteData("./output/"+filename+"_I.csv", I)
	WriteData("./output/"+filename+"_E.csv", E)
	WriteData("./output/"+filename+"_W.csv", W)
	WriteData("./output/"+filename+"_FROM.csv", FROM)
	WriteData("./output/"+filename+"_TO.csv", TO)
	fmt.Println("****************** ")
	fmt.Println("Summary 		: ")
	fmt.Println("Size V  		: ", len(V))
	fmt.Println("Size I  		: ", len(I))
	fmt.Println("Size E  		: ", len(E))
	fmt.Println("Size W		: ", len(W))
	fmt.Println("Size FROM		: ", len(FROM))
	fmt.Println("Size TO  		: ", len(TO))
	fmt.Println("****************** ")
	fmt.Println("Done !")
	return
}

func WriteData(filePath string, values []int) error {
	fmt.Println("Writing to ", filePath)
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()
	for id, value := range values {
		fmt.Fprint(f, value)
		if id < len(values)-1 {
			fmt.Fprint(f, ", ")
		}
	}
	fmt.Fprint(f, "\n")
	return nil
}

type Edge struct {
	u, v string
	w    int
}

func (e Edge) String() string {
	var str strings.Builder
	str.WriteString(e.u + " -> " + e.v + ", w = ")
	fmt.Fprintf(&str, "%d", e.w)
	str.WriteString("\n")
	return str.String()
}

type Graph struct {
	edges map[string][]Edge
	nodes HashSet
}

func (g Graph) String() string {
	var str strings.Builder
	str.WriteString("Graph {\n\tNodes : ")
	for key, _ := range g.nodes.elements {
		str.WriteString(key + " ")
	}
	str.WriteString("\n\tNumber of nodes : ")
	fmt.Fprintf(&str, "%d", g.NumNodes())
	str.WriteString("\n")
	str.WriteString("\tEdges : {\n")
	for _, edges := range g.edges {
		for _, edge := range edges {
			str.WriteString("\t\t" + edge.String())
		}
	}
	str.WriteString("\t}\n")
	str.WriteString("\tNumber of edges : ")
	fmt.Fprintf(&str, "%d\n", g.NumEdges())
	str.WriteString("}")
	return str.String()
}

func (g Graph) AddEdgeAndNodes(node string, e Edge) {
	if g.edges == nil {
		g.edges = make(map[string][]Edge)
	}
	g.nodes.add(e.u)
	g.nodes.add(e.v)
	g.edges[node] = append(g.edges[node], e)
}

func (g Graph) AddEdge(node string, e Edge) {
	if g.edges == nil {
		g.edges = make(map[string][]Edge)
	}
	g.edges[node] = append(g.edges[node], e)
}

func (g Graph) AddNode(node string) bool {
	return g.nodes.add(node)
}
func (g Graph) NumEdges() int {
	count := 0
	for _, edges := range g.edges {
		count += len(edges)
	}
	return count
}

func (g Graph) NumNodes() int {
	return g.nodes.NumElements()
}

func (g Graph) toString() {
	if g.edges == nil {
		return
	}
	for key := range g.edges {
		fmt.Printf(key + "\n")
	}
}

// HashSet
type HashSet struct {
	elements map[string]bool
}

func (set HashSet) add(elem string) bool {
	if _, found := set.elements[elem]; !found {
		set.elements[elem] = true
		return true
	} else {
		return false
	}
}

func (set HashSet) remove(elem string) bool {
	_, val := set.elements[elem]
	if val {
		delete(set.elements, elem)
		return true
	} else {
		return false
	}
}
func (set HashSet) NumElements() int {
	return len(set.elements)
}
func (set HashSet) print() {
	for elem := range set.elements {
		fmt.Printf(elem + "\n")
	}
}

// End HashSet
