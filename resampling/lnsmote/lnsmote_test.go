// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lnsmote_test

import (
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/knn"
	"github.com/shuLhan/go-mining/resampling/lnsmote"
	"github.com/shuLhan/tabula"
	"testing"
)

const (
	fcfg = "../../testdata/phoneme/phoneme.dsv"
)

func TestLNSmote(t *testing.T) {
	// Read sample dataset.
	dataset := tabula.Dataset{}
	_, e := dsv.SimpleRead(fcfg, &dataset)
	if nil != e {
		t.Fatal(e)
	}

	fmt.Println("[lnsmote_test] Total samples:", dataset.GetNRow())

	// write synthetic samples.
	writer, e := dsv.NewWriter("")

	if nil != e {
		t.Fatal(e)
	}

	e = writer.OpenOutput("phoneme_lnsmote.csv")
	if e != nil {
		t.Fatal(e)
	}

	writer.WriteRawRows(dataset.GetRows(), ",")

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

	synthetics := lnsmote.Resampling(&dataset)

	fmt.Println("[lnsmote_test] n synthetic:", synthetics.Len())

	writer.WriteRawRows(synthetics.GetRows(), ",")

	writer.Close()
}
