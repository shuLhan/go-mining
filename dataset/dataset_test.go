// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dataset_test

import (
	"io"
	"testing"

	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/dataset"
)

const (
	NRecords = 150
)

func TestReader (t *testing.T) {
	ds, e := dataset.NewReader ("../testdata/iris.dsv")

	if nil != e {
		t.Fatal (e)
	}

	_, e = dsv.Read (ds)

	if nil != e && e != io.EOF {
		t.Fatal (e)
	}

	if ds.GetRecordRead() != NRecords {
		t.Fatal("Dataset should be ", NRecords)
	}
}
