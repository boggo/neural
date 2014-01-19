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
	. "github.com/smartystreets/goconvey/convey"
	"math"
	"testing"
)

func TestNode(t *testing.T) {
	Convey("Subject: Nodes", t, func() {

		Convey("Given a new (internal) node", func() {
			var b, i, h, o node
			b = newNode(BIAS, DIRECT)
			i = newNode(INPUT, DIRECT)
			h = newNode(HIDDEN, DIRECT)
			o = newNode(OUTPUT, DIRECT)

			Convey("Input should default correctly", func() {
				Convey("BIAS should be 1.0", func() {
					So(b.input, ShouldEqual, 1.0)
				})
				Convey("INPUT should be 0.0", func() {
					So(i.input, ShouldEqual, 0.0)
				})
				Convey("HIDDEN should be 0.0", func() {
					So(h.input, ShouldEqual, 0.0)
				})
				Convey("OUTPUT should be 0.0", func() {
					So(o.input, ShouldEqual, 0.0)
				})
			})
			Convey("Node Type should be set correctly", func() {
				Convey("BIAS should be BIAS", func() {
					So(b.nodeType, ShouldEqual, BIAS)
					So(b.NodeType(), ShouldEqual, BIAS)
				})
				Convey("INPUT should be INPUT", func() {
					So(i.nodeType, ShouldEqual, INPUT)
					So(i.NodeType(), ShouldEqual, INPUT)
				})
				Convey("HIDDEN should be HIDDEN", func() {
					So(h.nodeType, ShouldEqual, HIDDEN)
					So(h.NodeType(), ShouldEqual, HIDDEN)
				})
				Convey("OUTPUT should be OUTPUT", func() {
					So(o.nodeType, ShouldEqual, OUTPUT)
					So(o.NodeType(), ShouldEqual, OUTPUT)
				})
			})
			Convey("Reset() should set Input back default value", func() {
				Convey("BIAS should be 1.0", func() {
					b.input = -1.0 // Set to a new value
					b.Reset()
					So(b.input, ShouldEqual, 1.0)
				})
				Convey("INPUT should be 0.0", func() {
					i.input = -1.0 // Set to a new value
					i.Reset()
					So(i.input, ShouldEqual, 0.0)
				})
				Convey("HIDDEN should be 0.0", func() {
					h.input = -1.0 // Set to a new value
					h.Reset()
					So(h.input, ShouldEqual, 0.0)
				})
				Convey("OUTPUT should be 0.0", func() {
					o.input = -1.0 // Set to a new value
					o.Reset()
					So(o.input, ShouldEqual, 0.0)
				})
			})
			Convey("Combine(x) should execute correctly base on Node Type", func() {
				Convey("BIAS should always be 1.0", func() {
					b.Reset()
					b.Combine(1.0) // Try to add a new value
					So(b.input, ShouldEqual, 1.0)
					b.Combine(-1.0)
					So(b.input, ShouldEqual, 1.0)
					b.Combine(-1.0)
					So(b.input, ShouldEqual, 1.0)
				})
				Convey("INPUT should replace input with x", func() {
					i.Reset()
					i.Combine(1.0) // Try to add a new value
					So(i.input, ShouldEqual, 1.0)
					i.Combine(-1.0)
					So(i.input, ShouldEqual, -1.0)
					i.Combine(0.0)
					So(i.input, ShouldEqual, 0.0)
				})
				Convey("HIDDEN should increment correctly", func() {
					h.Reset()
					h.Combine(1.0) // Try to add a new value
					So(h.input, ShouldEqual, 1.0)
					h.Combine(-1.0)
					So(h.input, ShouldEqual, 0.0)
					h.Combine(-1.0)
					So(h.input, ShouldEqual, -1.0)
				})
				Convey("OUTPUT should increment correctly", func() {
					o.Reset()
					o.Combine(1.0) // Try to add a new value
					So(o.input, ShouldEqual, 1.0)
					o.Combine(-1.0)
					So(o.input, ShouldEqual, 0.0)
					o.Combine(-1.0)
					So(o.input, ShouldEqual, -1.0)
				})
			})
		})

		Convey("Creating a new Node using NewNode function", func() {
			var node Node
			Convey("DIRECT should produce DirectNode", func() {
				node = NewNode(DIRECT, INPUT)
				So(node.FuncType(), ShouldEqual, DIRECT)
			})
			Convey("SIGMOID should produce SigmoidNode", func() {
				node = NewNode(SIGMOID, INPUT)
				So(node.FuncType(), ShouldEqual, SIGMOID)
			})
			Convey("STEEPENED_SIGMOID should produce SteependSigmoidNode", func() {
				node = NewNode(STEEPENED_SIGMOID, INPUT)
				So(node.FuncType(), ShouldEqual, STEEPENED_SIGMOID)
			})
		})

		Convey("Given a new Direct Node", func() {
			var b, i, h, o *DirectNode
			b = NewDirectNode(BIAS)
			i = NewDirectNode(INPUT)
			h = NewDirectNode(HIDDEN)
			o = NewDirectNode(OUTPUT)

			Convey("Function Type should be DIRECT", func() {
				So(b.funcType, ShouldEqual, DIRECT)
				So(i.funcType, ShouldEqual, DIRECT)
				So(h.funcType, ShouldEqual, DIRECT)
				So(o.funcType, ShouldEqual, DIRECT)

				So(b.FuncType(), ShouldEqual, DIRECT)
				So(i.FuncType(), ShouldEqual, DIRECT)
				So(h.FuncType(), ShouldEqual, DIRECT)
				So(o.FuncType(), ShouldEqual, DIRECT)
			})

			Convey("Activate() should return the appropriate value", func() {
				Convey("BIAS should always be 1.0", func() {
					b.Reset()
					b.Combine(1.0) // Try to add a new value
					So(b.Activate(), ShouldEqual, 1.0)
				})
				Convey("INPUT should return the same as input", func() {
					i.Reset()
					i.Combine(1.0) // Try to add a new value
					So(i.Activate(), ShouldEqual, 1.0)
				})
				Convey("HIDDEN should return the same as input", func() {
					h.Reset()
					h.Combine(1.0) // Try to add a new value
					So(h.Activate(), ShouldEqual, 1.0)
				})
				Convey("OUTPUT should return the same as input", func() {
					o.Reset()
					o.Combine(1.0) // Try to add a new value
					So(o.Activate(), ShouldEqual, 1.0)
				})
			})
		})

		Convey("Given a new Sigmoid Node", func() {
			var b, i, h, o *SigmoidNode
			b = NewSigmoidNode(BIAS)
			i = NewSigmoidNode(INPUT)
			h = NewSigmoidNode(HIDDEN)
			o = NewSigmoidNode(OUTPUT)

			Convey("Function Type should be SIGMOID", func() {
				So(b.funcType, ShouldEqual, SIGMOID)
				So(i.funcType, ShouldEqual, SIGMOID)
				So(h.funcType, ShouldEqual, SIGMOID)
				So(o.funcType, ShouldEqual, SIGMOID)

				So(b.FuncType(), ShouldEqual, SIGMOID)
				So(i.FuncType(), ShouldEqual, SIGMOID)
				So(h.FuncType(), ShouldEqual, SIGMOID)
				So(o.FuncType(), ShouldEqual, SIGMOID)
			})

			Convey("Activate() should return the appropriate value", func() {
				sig1 := 1.0 / (1.0 + math.Exp(-1.0))
				sig0 := 0.5

				Convey("BIAS should always be Sigmoid of 1.0", func() {
					b.Reset()
					So(b.Activate(), ShouldEqual, sig1)
					b.Combine(1.0) // Try to add a new value
					So(b.Activate(), ShouldEqual, sig1)
				})
				Convey("INPUT should return the correct Sigmoid value", func() {
					i.Reset()
					So(i.Activate(), ShouldEqual, sig0)
					i.Combine(1.0) // Try to add a new value
					So(i.Activate(), ShouldEqual, sig1)
				})
				Convey("HIDDEN should return the correct Sigmoid value", func() {
					h.Reset()
					So(h.Activate(), ShouldEqual, sig0)
					h.Combine(1.0) // Try to add a new value
					So(h.Activate(), ShouldEqual, sig1)
				})
				Convey("OUTPUT should return the correct Sigmoid value", func() {
					o.Reset()
					So(o.Activate(), ShouldEqual, sig0)
					o.Combine(1.0) // Try to add a new value
					So(o.Activate(), ShouldEqual, sig1)
				})
			})
		})

	})

}
