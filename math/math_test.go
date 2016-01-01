// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math_test

import (
	"testing"

	"github.com/shuLhan/go-mining/math"
)

func TestFactorial(t *testing.T) {
	in := []int{-3, -2, -1, 0, 1, 2, 3}
	exp := []int{-6, -2, -1, 1, 1, 2, 6}

	for i := range in {
		res := math.Factorial(in[i])

		if res != exp[i] {
			t.Fatal("Expecting ", exp[i], ", got ", res)
		}
	}
}

func TestBinomialCoefficient(t *testing.T) {
	in := [][]int{{1, 2}, {1, 1}, {3, 2}, {5, 3}}
	exp := []int{0, 1, 3, 10}

	for i := range in {
		res := math.BinomialCoefficient(in[i][0], in[i][1])

		if res != exp[i] {
			t.Fatal("Expecting ", exp[i], ", got ", res)
		}
	}
}

func TestStirlingS2(t *testing.T) {
	in := [][]int{{3, 1}, {3, 2}, {3, 3}, {5, 3}}
	exp := []int{1, 3, 1, 25}

	for i := range in {
		res := math.StirlingS2(in[i][0], in[i][1])

		if res != exp[i] {
			t.Fatal("Expecting ", exp[i], ", got ", res)
		}
	}
}
