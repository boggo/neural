neural
======

Neural Network in GO

I needed a Neural Network for my NEAT library written in GO. schulyer has a nice neural network
library in Go already (https://github.com/schuyler/neural-go) but it doesn't allow for the all the 
flexibility I need for NEAT. 

You can use this library in 2 ways. First, you can create a new Network simply by specifying the
number of each type of node:

```Go

numInputs := 2
numHidden := 3
numOutput := 1

network := neural.NewNetwork(numInputs, numHidden, numOutput)
inputs  := []float64 {1.234, -5.678}  
outputs := network.Activate(inputs)
```

This will create a new network including a bias node. The bias and inputs will be fully connected to
the hidden nodes. Likewise, the bias and hidden nodes will be full connected to the output nodes.


You can also build a network manually. This will allow you to select different activation functions 
for your nodes or to be more creative with how nodes are connected. To use this library in this manner,
first construct a few Nodes

```Go
bias := neural.NewNode(neural.DIRECT,  neural.BIAS)
in1  := neural.NewNode(neural.DIRECT,  neural.INPUT)
in2  := neural.NewNode(neural.DIRECT,  neural.INPUT)
hid1 := neural.NewNode(neural.SIGMOID, neural.HIDDEN)
out1 := neural.NewNode(neural.SIGMOID, neural.OUTPUT)
```

Alternatively, you could construct the same nodes by 

```Go
bias := neural.NewDirectNode(neural.BIAS)    
in1  := neural.NewDirectNode(neural.INPUT)    
in2  := neural.NewDirectNode(neural.INPUT)    
hid1 := neural.NewSigmoidNode(neural.HIDDEN)  
out1 := neural.NewSigmoidNode(neural.OUTPUT)
```

Then construct a few Connections

```Go
conn1 := neural.NewConnection(bias, hid1, rand.Float64() * 2 - 1)    
conn2 := neural.NewConnection(in1,  hid1, rand.Float64() * 2 - 1)  
conn3 := neural.NewConnection(in2,  hid1, rand.Float64() * 2 - 1)  
conn4 := neural.NewConnection(bias, out1, rand.Float64() * 2 - 1)  
conn5 := neural.NewConnection(hid1, out1, rand.Float64() * 2 - 1)
```

Then pull it all together

```Go
network := &neural.Network{}      // Create a empty Network

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
```

Finally, run the Network

```Go
inputs  := []float64 {1.234, -5.678}  
outputs := network.Activate(inputs)
```
