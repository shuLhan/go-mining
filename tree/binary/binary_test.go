// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package binary_test

import (
	"fmt"
	"testing"

	"github.com/shuLhan/go-mining/tree/binary"
)

func TestTree(t *testing.T) {
	exp := `1
	12
		24
			34
			33
		23
	11
		22
			32
			31
		21
`

	btree := binary.NewTree()

	root := binary.NewBTNode(1,
		binary.NewBTNode(11,
			binary.NewBTNode(21, nil, nil),
			binary.NewBTNode(22,
				binary.NewBTNode(31, nil, nil),
				binary.NewBTNode(32, nil, nil))),
		binary.NewBTNode(12,
			binary.NewBTNode(23, nil, nil),
			binary.NewBTNode(24,
				binary.NewBTNode(33, nil, nil),
				binary.NewBTNode(34, nil, nil))))

	btree.Root = root

	res := fmt.Sprint(btree)

	if res != exp {
		t.Fatal("error, expecting:\n", exp, "\n got:\n", res)
	}
}
