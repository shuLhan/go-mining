// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package smote_test

import (
	"fmt"
	"io"
	"testing"

	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/knn"
	"github.com/shuLhan/go-mining/resampling/smote"
)

var (
	fcfg        = "../../testdata/phoneme/phoneme.dsv"
	PercentOver = 100
	K           = 5
)

func doSmote(reader *dsv.Reader) (smot *smote.SMOTE) {
	smot = &smote.SMOTE{
		Input: knn.Input{
			DistanceMethod: knn.TEuclidianDistance,
			ClassIdx:       5,
			K:              K,
		},
		PercentOver: PercentOver,
		Synthetic:   nil,
	}

	classes := reader.Rows.GroupByValue(smot.ClassIdx)
	_, minClass := classes.GetMinority()

	fmt.Println("minority samples:", len(minClass))

	synthetic := smot.Resampling(minClass)

	fmt.Println("Synthetic:", len(synthetic))

	return smot
}

func TestSmote(t *testing.T) {
	var e error
	var n int
	var reader *dsv.Reader
	var writer *dsv.Writer

	reader, e = dsv.NewReader(fcfg)

	if nil != e {
		t.Fatal(e)
	}

	n, e = dsv.Read(reader)

	if nil != e && e != io.EOF {
		t.Fatal(e)
	}

	fmt.Println("Total samples:", n)

	// Open writer synthetic samples.
	writer, e = dsv.NewWriter("")

	if nil != e {
		t.Fatal(e)
	}

	e = writer.OpenOutput("phoneme_smote.csv")
	if e != nil {
		t.Fatal(e)
	}

	writer.WriteRawRows(&reader.Rows, ",")

	smot := doSmote(reader)

	writer.WriteRawRows(&smot.Synthetic, ",")

	reader.Close()
	writer.Close()
}
