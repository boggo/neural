package neural

import (
    "sort"
)

// A Network of Nodes and Connections
type Network struct {
    nodes nodeList
    conns connList

    biasCount   int
    inputCount  int
    outputCount int
    hiddenCount int
}

// Creates a new, empty Network
func NewNetwork() *Network {
    return &Network{}
}

// Adds a Node to the Network. The nodes are kept loosely sorted in order 
// of NodeType: Bias, Input, Output, Hidden
func (n *Network) AddNode(node Node) {

    // Add the node to the slice
    n.nodes = append(n.nodes, node)
    sort.Sort(n.nodes)

    // Update the internal counts
    switch node.NodeType() {
    case BIAS:
        n.biasCount++
    case INPUT:
        n.inputCount++
    case OUTPUT:
        n.outputCount++
    case HIDDEN:
        n.hiddenCount++
    }
}

// Adds a Connection to the Network. Connections should be added in the order
// in which they should be activated
func (n *Network) AddConnection(conn Connection) {
    n.conns = append(n.conns, conn)
}

// Activates the Network. Takes a slice of float64 values as input and outputs
// a slice of float64 values. Note: The network is updated during this method.
func (n *Network) Activate(inputs []float64) (outputs []float64) {

    // Reset the network
    for i, _ := range n.nodes {
        n.nodes[i].Reset()
    }

    // Calculate the offsets
    inputOffset := n.biasCount
    outputOffset := inputOffset + n.inputCount

    // Set the inputs
    for i, _ := range inputs {
        n.nodes[i+inputOffset].Combine(inputs[i])
    }

    // Activate all the connections
    for i, _ := range n.conns {
        n.conns[i].activate()
    }

    // Return the outputs
    outputs = make([]float64, n.outputCount)
    for i := 0; i < n.outputCount; i++ {
        outputs[i] = n.nodes[i+outputOffset].Activate()
    }
    return
}
