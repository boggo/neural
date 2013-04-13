package neural

// Connection interface
type Connection interface {
    activate()
}

// Implementation of Connection as a private package struct
type connection struct {
    fromNode Node
    toNode   Node
    weight   float64
}

// List of Connections
type connList []Connection

// Creates a new Connection
func NewConnection(fromNode Node, toNode Node, weight float64) Connection {
    return &connection{fromNode, toNode, weight}
}

// Activates a connection by taking the activation of the source node, 
// multiplying it by the connection weight and combining that with the
// value of the target node
func (c *connection) activate() {
    c.toNode.Combine(c.fromNode.Activate() * c.weight)
}
