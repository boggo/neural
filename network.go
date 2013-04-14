package neural

import (
    "math/rand"
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
func NewNetwork(numInput, numHidden, numOutput int) *Network {

    network := &Network{}

    // Add the bias node
    network.AddNode(NewNode(DIRECT, BIAS))

    // Add the input nodes
    for i := 0; i < numInput; i++ {
        network.AddNode(NewNode(DIRECT, INPUT))
    }

    // Add the hidden nodes
    for i := 0; i < numHidden; i++ {
        network.AddNode(NewNode(SIGMOID, HIDDEN))
    }

    // Add the output nodes
    for i := 0; i < numOutput; i++ {
        network.AddNode(NewNode(SIGMOID, OUTPUT))
    }

    // Node the offsets
    inputOffset := network.biasCount
    outputOffset := inputOffset + network.inputCount
    hiddenOffset := outputOffset + network.outputCount

    // Connect to the input layer to the hidden layer
    for h := 0; h < network.hiddenCount; h++ {

        // Connect to the bias node
        network.AddConnection(NewConnection(network.nodes[0], network.nodes[hiddenOffset+h], rand.Float64()*2-1))

        // Connect to the input nodes
        for i := 0; i < network.inputCount; i++ {
            network.AddConnection(NewConnection(network.nodes[inputOffset+i], network.nodes[hiddenOffset+h], rand.Float64()*2-1))
        }
    }

    // Connect the hidden layer to the output layer
    for o := 0; o < network.outputCount; o++ {

        // Connect to the bias node
        network.AddConnection(NewConnection(network.nodes[0], network.nodes[outputOffset+o], rand.Float64()*2-1))

        // Connect to the hidden nodes
        for h := 0; h < network.hiddenCount; h++ {
            network.AddConnection(NewConnection(network.nodes[hiddenOffset+h], network.nodes[outputOffset+o], rand.Float64()*2-1))
        }
    }
    // Return the network
    return network
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
