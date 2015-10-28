// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gini_test

import (
	"fmt"
	"testing"

	"github.com/shuLhan/go-mining/gain/gini"
)

var data = []float64{ 1.0, 6.0, 5.0, 4.0, 7.0, 3.0, 8.0, 7.0, 5.0 }
var target = []string{ "P", "P", "N", "P", "N", "N", "N", "P", "N" }
var classes = []string { "P", "N" }

func TestComputeContinu(t *testing.T) {
	gini := gain.Gini{}

	fmt.Println ("target:", target)
	fmt.Println ("data:", data)

	gini.ComputeContinu(&data, &target, &classes)

	fmt.Println ("sorted index:", gini.SortedIndex)
	fmt.Println ("Gini value:", gini.Value)
	fmt.Println ("Gini partition:", gini.Part)
	fmt.Println ("Gini index:", gini.Index)
	fmt.Println ("Gini max partition value:", gini.GetMaxPartValue())
	fmt.Println ("Gini max gain:", gini.GetMaxGainValue())
}
