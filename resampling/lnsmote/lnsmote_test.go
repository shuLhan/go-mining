// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lnsmote_test

import (
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/knn"
	"github.com/shuLhan/go-mining/resampling/lnsmote"
	"io"
	"testing"
)

const (
	fcfg = "../../testdata/phoneme/phoneme.dsv"
)

func TestLNSmote(t *testing.T) {
	// Read sample dataset.
	reader, e := dsv.NewReader(fcfg)

	if nil != e {
		t.Fatal(e)
	}

	n, e := dsv.Read(reader)

	if nil != e && e != io.EOF {
		t.Fatal(e)
	}

	reader.Close()

	fmt.Println("[lnsmote_test] Total samples:", n)

	// write synthetic samples.
	writer, e := dsv.NewWriter("")

	if nil != e {
		t.Fatal(e)
	}

	e = writer.OpenOutput("phoneme_lnsmote.csv")
	if e != nil {
		t.Fatal(e)
	}

	writer.WriteRawRows(&reader.Rows, ",")

	// Initialize LN-SMOTE.
	lnsmote := &lnsmote.Input{
		Input: knn.Input{
			DistanceMethod: knn.TEuclidianDistance,
			ClassIdx:       5,
			K:              5,
		},
		ClassMinor:  "1",
		PercentOver: 100,
	}

	synthetics := lnsmote.Resampling(reader.Dataset)

	fmt.Println("[lnsmote_test] n synthetic:", synthetics.Len())

	writer.WriteRawRows(&synthetics.Rows, ",")

	writer.Close()
}
