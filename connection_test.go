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
	"testing"
)

func TestConnection(t *testing.T) {
	Convey("Subject: Connection", t, func() {
		var src, tgt *DirectNode
		src = NewDirectNode(INPUT)
		tgt = NewDirectNode(OUTPUT)
		Convey("Given a new Connection", func() {
			con := NewConnection(src, tgt, 0.5)
			Convey("From Node should equal Source", func() {
				So(con.fromNode, ShouldEqual, src)
			})
			Convey("To Node should equal Target", func() {
				So(con.toNode, ShouldEqual, tgt)
			})
			Convey("Weight should be set correctly", func() {
				So(con.weight, ShouldEqual, 0.5)
			})
			Convey("Activation should work correctly", func() {
				src.input = 0.5
				con.weight = 0.5
				con.activate()
				So(tgt.input, ShouldEqual, 0.25)
			})
		})
	})
}
