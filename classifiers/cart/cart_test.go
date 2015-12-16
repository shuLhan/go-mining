// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cart_test

import (
	"fmt"
	"github.com/shuLhan/dsv/util/assert"
	"github.com/shuLhan/go-mining/classifiers/cart"
	"github.com/shuLhan/go-mining/dataset"
	"io"
	"testing"
)

const (
	NRows = 150
)

func TestCART(t *testing.T) {
	ds, e := dataset.NewReader("../../testdata/iris/iris.dsv")

	if nil != e {
		t.Fatal(e)
	}

	e = ds.Read()

	if nil != e && e != io.EOF {
		t.Fatal(e)
	}

	assert.Equal(t, NRows, ds.GetNRow())

	// Build CART tree.
	CART := cart.NewInput(cart.SplitMethodGini)

	e = CART.BuildTree(ds)

	if e != nil {
		t.Fatal(e)
	}

	fmt.Println("CART Tree:\n", CART)

	// Reread the data
	ds.Reset()
	ds.Open()

	e = ds.Read()
	if nil != e && e != io.EOF {
		t.Fatal(e)
	}

	// Create test set
	testset, e := dataset.NewReader("../../testdata/iris/iris.dsv")

	if nil != e {
		t.Fatal(e)
	}

	e = testset.Read()

	if nil != e && e != io.EOF {
		t.Fatal(e)
	}

	testset.GetTarget().ClearValues()

	// Classifiy test set
	CART.ClassifySet(testset)

	assert.Equal(t, ds.GetTarget(), testset.GetTarget())
}
