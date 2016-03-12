// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package smote_test

import (
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/knn"
	"github.com/shuLhan/go-mining/resampling/smote"
	"github.com/shuLhan/tabula"
	"testing"
)

var (
	fcfg        = "../../testdata/phoneme/phoneme.dsv"
	PercentOver = 100
	K           = 5
)

func doSmote(dataset tabula.DatasetInterface) (smot *smote.SMOTE) {
	smot = &smote.SMOTE{
		Input: knn.Input{
			DistanceMethod: knn.TEuclidianDistance,
			ClassIdx:       5,
			K:              K,
		},
		PercentOver: PercentOver,
		Synthetic:   nil,
	}

	classes := dataset.GetRows().GroupByValue(smot.ClassIdx)
	_, minClass := classes.GetMinority()

	fmt.Println("minority samples:", len(minClass))

	synthetic := smot.Resampling(minClass)

	fmt.Println("Synthetic:", len(synthetic))

	return smot
}

func TestSmote(t *testing.T) {
	dataset := tabula.Dataset{}

	_, e := dsv.SimpleRead(fcfg, &dataset)

	if nil != e {
		t.Fatal(e)
	}

	fmt.Println("Total samples:", dataset.Len())

	// Open writer synthetic samples.
	writer, e := dsv.NewWriter("")

	if nil != e {
		t.Fatal(e)
	}

	e = writer.OpenOutput("phoneme_smote.csv")
	if e != nil {
		t.Fatal(e)
	}

	writer.WriteRawRows(dataset.GetRows(), ",")

	smot := doSmote(&dataset)

	writer.WriteRawRows(&smot.Synthetic, ",")

	writer.Close()
}
