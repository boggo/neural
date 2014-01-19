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
	"github.com/boggo/random"
	. "github.com/smartystreets/goconvey/convey"
	"math"
	"testing"
)

func TestNetwork(t *testing.T) {
	Convey("Subject: Network", t, func() {
		random.Reseed(0) // Get a predictable random number generation
		Convey("Given a new, empty Network", func() {
			var net *Network
			var bias, in1, in2, hid1, hid2, out1, out2 Node
			net = &Network{}
			Convey("The network should be empty", func() {
				So(len(net.nodes), ShouldEqual, 0)
				So(len(net.conns), ShouldEqual, 0)
				So(net.biasCount, ShouldEqual, 0)
				So(net.inputCount, ShouldEqual, 0)
				So(net.outputCount, ShouldEqual, 0)
				So(net.hiddenCount, ShouldEqual, 0)
			})
			Convey("Adding a nodes should change the structure", func() {
				Convey("Adding a BIAS node is successful", func() {
					bias = NewDirectNode(BIAS)
					net.AddNode(bias)
					So(net.biasCount, ShouldEqual, 1)
					So(net.inputCount, ShouldEqual, 0)
					So(net.outputCount, ShouldEqual, 0)
					So(net.hiddenCount, ShouldEqual, 0)
					So(len(net.nodes), ShouldEqual, 1)
				})
				Convey("Adding INPUT nodes is successful", func() {
					in1 = NewDirectNode(INPUT)
					in2 = NewDirectNode(INPUT)
					net.AddNode(in1)
					net.AddNode(in2)
					So(net.biasCount, ShouldEqual, 1)
					So(net.inputCount, ShouldEqual, 2)
					So(net.outputCount, ShouldEqual, 0)
					So(net.hiddenCount, ShouldEqual, 0)
					So(len(net.nodes), ShouldEqual, 3)
				})
				Convey("Adding HIDDEN nodes is successful", func() {
					hid1 = NewSigmoidNode(HIDDEN)
					hid2 = NewSigmoidNode(HIDDEN)
					net.AddNode(hid1)
					net.AddNode(hid2)
					So(net.biasCount, ShouldEqual, 1)
					So(net.inputCount, ShouldEqual, 2)
					So(net.outputCount, ShouldEqual, 0)
					So(net.hiddenCount, ShouldEqual, 2)
					So(len(net.nodes), ShouldEqual, 5)
				})
				Convey("Adding OUTPUT nodes is successful", func() {
					out1 = NewSigmoidNode(OUTPUT)
					out2 = NewSigmoidNode(OUTPUT)
					net.AddNode(out1)
					net.AddNode(out2)
					So(net.biasCount, ShouldEqual, 1)
					So(net.inputCount, ShouldEqual, 2)
					So(net.outputCount, ShouldEqual, 2)
					So(net.hiddenCount, ShouldEqual, 2)
					So(len(net.nodes), ShouldEqual, 7)
				})
			})
			Convey("Adding connections should change the structure", func() {
				net.AddConnection(NewConnection(bias, hid1, 0.1))
				net.AddConnection(NewConnection(bias, hid2, 0.2))
				net.AddConnection(NewConnection(bias, out1, 0.3))
				net.AddConnection(NewConnection(bias, out2, 0.4))
				net.AddConnection(NewConnection(in1, hid1, 0.5))
				net.AddConnection(NewConnection(in1, hid2, 0.6))
				net.AddConnection(NewConnection(in2, hid1, 0.7))
				net.AddConnection(NewConnection(in2, hid2, 0.8))
				net.AddConnection(NewConnection(hid1, out1, 0.9))
				net.AddConnection(NewConnection(hid1, out2, 1.0))
				net.AddConnection(NewConnection(hid2, out1, 1.1))
				net.AddConnection(NewConnection(hid2, out2, 1.2))
				So(len(net.conns), ShouldEqual, 12)
			})
			Convey("Activation should produce the correct results", func() {
				inputs := []float64{0.25, 0.75}
				Convey("First pass should work fine", func() {
					outputs := net.Activate(inputs)
					So(math.Floor(outputs[0]*10000.0), ShouldEqual, 8461)
					So(math.Floor(outputs[1]*10000.0), ShouldEqual, 8748)
				})
				Convey("Second pass should produce the same result", func() {
					outputs := net.Activate(inputs)
					So(math.Floor(outputs[0]*10000.0), ShouldEqual, 8461)
					So(math.Floor(outputs[1]*10000.0), ShouldEqual, 8748)
				})
			})
		})
		Convey("Given a new Network specification", func() {
			Convey("It should have the right number of components", func() {
				net := NewNetwork(2, 2, 2)
				So(len(net.nodes), ShouldEqual, 7)
				So(len(net.conns), ShouldEqual, 12)
				So(net.biasCount, ShouldEqual, 1)
				So(net.inputCount, ShouldEqual, 2)
				So(net.outputCount, ShouldEqual, 2)
				So(net.hiddenCount, ShouldEqual, 2)
			})
		})
	})
}
