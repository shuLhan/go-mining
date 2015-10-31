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
var targetValues = []string{ "P", "P", "N", "P", "N", "N", "N", "P", "N" }
var classes = []string { "P", "N" }

func TestComputeContinu(t *testing.T) {
	gini := gini.Gini{}
	target := make([]string, len(targetValues))

	copy(target, targetValues)

	fmt.Println ("target:", target)
	fmt.Println ("data:", data)

	gini.ComputeContinu(&data, &target, &classes)

	fmt.Println (gini)
	fmt.Println ("Gini max partition value:", gini.GetMaxPartValue())
	fmt.Println ("Gini max gain:", gini.GetMaxGainValue())
}

var discreteSamples = []string{ "T", "T", "T", "F", "F", "F", "F", "T", "F"}
var discreteValues = []string{ "T", "F" }

func TestComputeDiscrete(t *testing.T) {
	gini := gini.Gini{}
	target := make([]string, len(targetValues))

	copy(target, targetValues)

	fmt.Println ("target:", target)
	fmt.Println ("data:", discreteSamples)

	gini.ComputeDiscrete(&discreteSamples, &discreteValues, &target,
				&classes)

	fmt.Println (gini)
	fmt.Println ("Gini max gain:", gini.GetMaxGainValue())
}
