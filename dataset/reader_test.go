// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dataset_test

import (
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/dsv/util/assert"
	"github.com/shuLhan/go-mining/dataset"
	"io"
	"testing"
)

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

	assert.Equal(t, NRecords, ds.GetNRow())

	// Split the iris-setosa
	setosa, _, e := ds.SplitRowsByValue(4, []string{ClassName})

	if e != nil {
		t.Fatal(e)
	}

	singleClass, name := ds.IsInSingleClass()
	assert.Equal(t, false, singleClass)
	assert.Equal(t, "", name)

	singleClass, name = setosa.IsInSingleClass()
	assert.Equal(t, true, singleClass)
	assert.Equal(t, ClassName, name)
}
