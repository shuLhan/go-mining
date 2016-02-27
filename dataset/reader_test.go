// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dataset_test

import (
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/dataset"
	"io"
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

const (
	NRecords  = 150
	ClassName = "Iris-setosa"
)

func TestReader(t *testing.T) {
	ds, e := dataset.NewReader("../testdata/iris/iris.dsv")

	if nil != e {
		t.Fatal(e)
	}

	_, e = dsv.Read(ds)

	if nil != e && e != io.EOF {
		t.Fatal(e)
	}

	assert(t, NRecords, ds.GetNRow(), true)

	// Split the iris-setosa
	setosa, _, e := ds.SplitRowsByValue(4, []string{ClassName})

	if e != nil {
		t.Fatal(e)
	}

	singleClass, name := ds.IsInSingleClass()
	assert(t, false, singleClass, true)
	assert(t, "", name, true)

	singleClass, name = setosa.IsInSingleClass()
	assert(t, true, singleClass, true)
	assert(t, ClassName, name, true)
}
