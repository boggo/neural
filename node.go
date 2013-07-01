package neural

import (
    "math"
)

// NodeType to distiguish Nodes
type NodeType byte

// Constants for NodeTypes
const (
    BIAS NodeType = iota
    INPUT
    OUTPUT
    HIDDEN
)

// FuncType to identify activation function
type FuncType byte

const (
    DIRECT FuncType = iota
    SIGMOID
	STEEPENED_SIGMOID
)

var (
	FuncTypes = []FuncType {DIRECT, SIGMOID, STEEPENED_SIGMOID}
)

// Node interface
type Node interface {
    Reset()
    Combine(value float64)
    Activate() float64
    NodeType() NodeType
    FuncType() FuncType
}

// List of Nodes
type nodeList []Node

// Len is the number of Nodes in the nodeList. Part of sort.Interface
func (nl nodeList) Len() int { return len(nl) }

// Less returns whether the Node with index i in the nodeList should
// sort before Node with index j. Part of sort.Interface
func (nl nodeList) Less(i, j int) bool { return nl[i].NodeType() < nl[j].NodeType() }

// Swap swaps the Nodes with indexes i and j. Part of sort.Interface
func (nl nodeList) Swap(i, j int) { nl[i], nl[j] = nl[j], nl[i] }

// node is the default implementation of Node as a private package struct
type node struct {
    input    float64
    nodeType NodeType
    funcType FuncType
}

// Reset a node to its starting value, 0. For Bias nodes, this is 1.
func (n *node) Reset() {
    if n.NodeType() == BIAS {
        n.input = 1
    } else {
        n.input = 0
    }
}

// Combines the new value with the existing input value of the Node. For Bias
// nodes this throws an error. For input nodes, this just replaces the input value.
// For all other Nodes, this adds the new value to the input value.
func (n *node) Combine(value float64) {
    switch n.NodeType() {
    case BIAS:
        //throw error. BIAS is always 1
    case INPUT:
        n.input = value
    default:
        n.input += value
    }
}

// NodeType returns the NodeType of the Node
func (n node) NodeType() NodeType {
    return n.nodeType
}

// FuncType returns the type of activation function
func (n node) FuncType() FuncType {
    return n.funcType
}

// NewNode returns the appropriate Node based on FuncType
func NewNode(funcType FuncType, nodeType NodeType) Node {
    switch funcType {
    case DIRECT:
        return NewDirectNode(nodeType)
    case SIGMOID:
        return NewSigmoidNode(nodeType)
	case STEEPENED_SIGMOID:
		return NewSteepenedSigmoidNode(nodeType)
    }

    // Unknown FuncType, return nil
    return nil
}

// DirectNode is an implemenation of Node which returns its input value without
// transformation
type DirectNode struct {
    node
}

// NewDirectNode returns a pointer to a new DirectNode
func NewDirectNode(nodeType NodeType) *DirectNode {
    return &DirectNode{node{nodeType: nodeType}}
}

// Activate returns the input value without transformation
func (n DirectNode) Activate() float64 {
    return n.input
}

// SigmoidNode is an implementation of Node which returns its input value transformed
// by the sigmoid function.
type SigmoidNode struct {
    node
}

// NewSigmoidNode returns a pointer to a new Sigmoid Node
func NewSigmoidNode(nodeType NodeType) *SigmoidNode {
    return &SigmoidNode{node{nodeType: nodeType}}
}

// Activate returns the input value transformed by the Sigmoid function:
// by the sigmoid function:    1
//                          -------
//                               -t
//                          1 + e
func (n SigmoidNode) Activate() float64 {
    return 1.0 / (1.0 + math.Pow(math.E, -n.input))
}

// SigmoidNode is an implementation of Node which returns its input value transformed
// by the sigmoid function.
type SteepenedSigmoidNode struct {
    node
}

// NewSigmoidNode returns a pointer to a new Sigmoid Node
func NewSteepenedSigmoidNode(nodeType NodeType) *SteepenedSigmoidNode {
    return &SteepenedSigmoidNode{node{nodeType: nodeType}}
}

// Activate returns the input value transformed by the Sigmoid function:
// by the sigmoid function:    1
//                          -------
//                               -4.9t
//                          1 + e
func (n SteepenedSigmoidNode) Activate() float64 {
    return 1.0 / (1.0 + math.Pow(math.E, -4.9*n.input))
}
