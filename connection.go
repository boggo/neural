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
)

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

func (c *connection) String() string {
	return fmt.Sprintf("%v, %v, %v", c.weight, c.fromNode.NodeType(), c.toNode.NodeType())
}
