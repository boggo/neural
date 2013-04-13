neural
======

Neural Network in GO

I needed a Neural Network for my NEAT library written in GO. schulyer has a nice neural network
library in Go already (https://github.com/schuyler/neural-go) but it doesn't allow for the all the 
flexibility I need for NEAT. 

To use this library, first construct a few Nodes

bias := neural.NewDirectNode(neural.BIAS)
in1  := neural.NewDirectNode(neural.INPUT)
in2  := neural.NewDirectNode(neural.INPUT)
hid1 := neural.NewSigmoidNode(neural.HIDDEN)
out1 := neural.NewSigmoidNode(neural.OUTPUT)

Then construct a few Connections

conn1 := neural.NewConnection(bias, hid1, rand.Float64() * 2 - 1)
conn2 := neural.NewConnection(in1,  hid1, rand.Float64() * 2 - 1)
conn3 := neural.NewConnection(in2,  hid1, rand.Float64() * 2 - 1)

conn4 := neural.NewConnection(bias, out1, rand.Float64() * 2 - 1)
conn5 := neural.NewConnection(hid1, out1, rand.Float64() * 2 - 1)

Then pull it all together

network := neural.NewNetwork()

network.AddNode(bias)
network.AddNode(in1)
network.AddNode(in2)
network.AddNode(hid1)
network.AddNode(out1)

network.AddConnection(conn1)			// It is important to add the connections
network.AddConnection(conn2)			// in the order they should execute
network.AddConnection(conn3)
network.AddConnection(conn4)
network.AddConnection(conn5)

Finally, run the Network
inputs := []float64 {1.234, -5.678}
outpus := network.Activate(inputs)
