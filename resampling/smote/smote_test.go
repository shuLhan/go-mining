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

const (
	fcfg = "../../testdata/phoneme/phoneme.dsv"
)

func doSmote(reader *dsv.Reader) (smot *smote.SMOTE, e error) {
	smot = &smote.SMOTE{
		Input: knn.Input{
			Dataset:        nil,
			DistanceMethod: knn.TEuclidianDistance,
			ClassIdx:       5,
			K:              5,
		},
		PercentOver: 200,
		Synthetic:   nil,
	}

	classes := reader.Rows.GroupByValue(smot.ClassIdx)
	_, minClass := classes.GetMinority()

	fmt.Println("minority samples:", len(minClass))

	smot.Dataset = minClass
	synthetic, e := smot.Resampling()

	if e != nil {
		return nil, e
	}

	fmt.Println("Synthetic:", len(synthetic))

	return smot, e
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

	smot, e := doSmote(reader)

	if e != nil {
		t.Fatal(e)
	}

	reader.Close()

	// write synthetic samples.
	writer, e = dsv.NewWriter(fcfg)

	if nil != e {
		t.Fatal(e)
	}

	writer.WriteRows(smot.Synthetic, reader.GetInputMetadata())

	writer.Close()
}
