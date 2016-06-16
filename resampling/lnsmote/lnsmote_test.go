// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lnsmote_test

import (
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/resampling/lnsmote"
	"github.com/shuLhan/tabula"
	"testing"
)

const (
	fcfg = "../../testdata/phoneme/phoneme.dsv"
)

func TestLNSmote(t *testing.T) {
	// Read sample dataset.
	dataset := tabula.Claset{}
	_, e := dsv.SimpleRead(fcfg, &dataset)
	if nil != e {
		t.Fatal(e)
	}

	fmt.Println("[lnsmote_test] Total samples:", dataset.GetNRow())

	// Write original samples.
	writer, e := dsv.NewWriter("")

	if nil != e {
		t.Fatal(e)
	}

	e = writer.OpenOutput("phoneme_lnsmote.csv")
	if e != nil {
		t.Fatal(e)
	}

	sep := dsv.DefSeparator
	_, e = writer.WriteRawRows(dataset.GetRows(), &sep)
	if e != nil {
		t.Fatal(e)
	}

	// Initialize LN-SMOTE.
	lnsmoteRun := lnsmote.New(100, 5, 5, "1", "lnsmote.outliers")

	e = lnsmoteRun.Resampling(&dataset)

	fmt.Println("[lnsmote_test] # synthetic:", lnsmoteRun.Synthetics.Len())

	sep = dsv.DefSeparator
	_, e = writer.WriteRawRows(lnsmoteRun.Synthetics.GetRows(), &sep)
	if e != nil {
		t.Fatal(e)
	}

	e = writer.Close()
	if e != nil {
		t.Fatal(e)
	}
}
