// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cart_test

import (
	"fmt"
	"io"
	"testing"

	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/dataset"
	"github.com/shuLhan/go-mining/classifiers/cart"
)

const (
	NRecords = 150
)

func TestCART(t *testing.T) {
	ds, e := dataset.NewReader("../../testdata/iris.dsv")

	if nil != e {
		t.Fatal(e)
	}

	_, e = dsv.Read(ds)

	if nil != e && e != io.EOF {
		t.Fatal(e)
	}

	if ds.GetRecordRead() != NRecords {
		t.Fatal("Dataset should be ", NRecords)
	}

	D, e := dataset.NewInput(&ds.Fields, &ds.InputMetadata, ds.ClassIndex)

	// Build CART tree.
	CART := cart.NewInput(cart.SplitMethodGini)

	CART.BuildTree(D)

	fmt.Println("CART Tree:\n", CART)
}
