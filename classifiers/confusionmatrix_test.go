// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package classifiers_test

import (
	"fmt"
	"github.com/shuLhan/go-mining/classifiers"
	"reflect"
	"runtime/debug"
	"testing"
)

func assert(t *testing.T, exp, got interface{}, equal bool) {
	if reflect.DeepEqual(exp, got) != equal {
		debug.PrintStack()
		t.Fatalf("\n"+
			">>> Expecting '%v'\n"+
			"          got '%v'\n", exp, got)
	}
}

func TestComputeNumeric(t *testing.T) {
	actuals := []int64{1, 1, 1, 0, 0, 0, 0}
	predics := []int64{1, 1, 0, 0, 0, 0, 1}
	vs := []int64{1, 0}
	exp := []int{2, 1, 3, 1}

	cm := &classifiers.ConfusionMatrix{}

	cm.ComputeNumeric(vs, actuals, predics)

	assert(t, exp[0], cm.TP(), true)
	assert(t, exp[1], cm.FN(), true)
	assert(t, exp[2], cm.TN(), true)
	assert(t, exp[3], cm.FP(), true)

	fmt.Println(cm)
}

func TestComputeStrings(t *testing.T) {
	actuals := []string{"1", "1", "1", "0", "0", "0", "0"}
	predics := []string{"1", "1", "0", "0", "0", "0", "1"}
	vs := []string{"1", "0"}
	exp := []int{2, 1, 3, 1}

	cm := &classifiers.ConfusionMatrix{}

	cm.ComputeStrings(vs, actuals, predics)

	assert(t, exp[0], cm.TP(), true)
	assert(t, exp[1], cm.FN(), true)
	assert(t, exp[2], cm.TN(), true)
	assert(t, exp[3], cm.FP(), true)

	fmt.Println(cm)
}

func TestGroupIndexPredictions(t *testing.T) {
	testIds := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	actuals := []int64{1, 1, 1, 1, 0, 0, 0, 0, 0, 0}
	predics := []int64{1, 1, 0, 1, 0, 0, 0, 0, 1, 0}
	exp := [][]int{
		{0, 1, 3},       // tp
		{2},             // fn
		{8},             // fp
		{4, 5, 6, 7, 9}, // tn
	}

	cm := &classifiers.ConfusionMatrix{}

	cm.GroupIndexPredictions(testIds, actuals, predics)

	assert(t, exp[0], cm.TPIndices(), true)
	assert(t, exp[1], cm.FNIndices(), true)
	assert(t, exp[2], cm.FPIndices(), true)
	assert(t, exp[3], cm.TNIndices(), true)
}
