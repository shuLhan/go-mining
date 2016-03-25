// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cart_test

import (
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/classifiers/cart"
	"github.com/shuLhan/tabula"
	"reflect"
	"runtime/debug"
	"testing"
)

const (
	NRows = 150
)

func assert(t *testing.T, exp, got interface{}, equal bool) {
	if reflect.DeepEqual(exp, got) != equal {
		debug.PrintStack()
		t.Fatalf("\n"+
			">>> Expecting '%v'\n"+
			"          got '%v'\n", exp, got)
	}
}

func TestCART(t *testing.T) {
	fds := "../../testdata/iris/iris.dsv"

	ds := tabula.Claset{}

	_, e := dsv.SimpleRead(fds, &ds)
	if nil != e {
		t.Fatal(e)
	}

	fmt.Println("[cart_test] class index:", ds.GetClassIndex())

	// copy target to be compared later.
	targetv := ds.GetClassAsStrings()

	assert(t, NRows, ds.GetNRow(), true)

	// Build CART tree.
	CART, e := cart.New(&ds, cart.SplitMethodGini, 0)
	if e != nil {
		t.Fatal(e)
	}

	fmt.Println("[cart_test] CART Tree:\n", CART)

	// Create test set
	testset := tabula.Claset{}
	_, e = dsv.SimpleRead(fds, &testset)

	if nil != e {
		t.Fatal(e)
	}

	testset.GetClassColumn().ClearValues()

	// Classifiy test set
	e = CART.ClassifySet(&testset)
	if nil != e {
		t.Fatal(e)
	}

	assert(t, targetv, testset.GetClassAsStrings(), true)
}
