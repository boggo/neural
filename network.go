/*  Copyright (c) 2013, Brian Hummer (brian@boggo.net)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the boggo.net nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL BRIAN HUMMER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

package neural

import (
	"fmt"
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

func (n *Network) Dump() {

	// Show the nodes
	for i, x := range n.nodes {
		fmt.Printf("[%d] %v, %v\n", i, x.NodeType(), x.FuncType())
	}

	// Show the connections
	for i, x := range n.conns {
		fmt.Println(i, x)
	}
}
