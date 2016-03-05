// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package randomforest_test

import (
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/classifiers/randomforest"
	"github.com/shuLhan/go-mining/dataset"
	"github.com/shuLhan/tabula"
	"io"
	"testing"
)

var (
	// NTREE number of tree to generate.
	NTREE = 10
	// NBOOTSTRAP percentage of sample used as subsample.
	NBOOTSTRAP = 66
	// FEATSTART number of feature to begin with.
	FEATSTART = 1
	// FEATEND maximum number of feature to test
	FEATEND = -1
)

func runRandomForest(t *testing.T, sampledsv string,
	ntree, npercent, nStart, nEnd int,
	oobFile string,
) {
	// read data.
	samples, e := dataset.NewReader(sampledsv)

	if nil != e {
		t.Fatal(e)
	}

	defer samples.Close()

	e = samples.Read()

	if nil != e && e != io.EOF {
		t.Fatal(e)
	}

	// dataset to save each oob error in each feature iteration.
	dataooberr := tabula.NewDataset(tabula.DatasetModeColumns, nil, nil)

	if nEnd < 0 {
		nEnd = samples.GetNColumn()
	}

	for ; nStart < nEnd; nStart++ {
		// create random forest.
		forest := randomforest.New(ntree, nStart, npercent)

		e := forest.Build(samples)

		if e != nil {
			t.Fatal(e)
		}

		colName := fmt.Sprintf("M%d", nStart)

		col := tabula.NewColumnReal(forest.OobErrSteps, colName)

		dataooberr.PushColumn(*col)

		if randomforest.RANDOMFOREST_DEBUG >= 3 {
			fmt.Println("[randomforest_test] ", forest)
		}
	}

	// write oob data to file
	writer, e := dsv.NewWriter("")

	if e != nil {
		t.Fatal(e)
	}

	e = writer.OpenOutput(oobFile)

	if e != nil {
		t.Fatal(e)
	}

	_, e = writer.WriteRawDataset(dataooberr, nil)
	if e != nil {
		t.Fatal(e)
	}
}

func TestEnsemblingGlass(t *testing.T) {
	sampledsv := "../../testdata/forensic_glass/fgl.dsv"
	// oob file output
	oobFile := "oobglass"

	runRandomForest(t, sampledsv, NTREE, NBOOTSTRAP, FEATSTART, FEATEND,
		oobFile)
}

func TestEnsemblingIris(t *testing.T) {
	// input data
	sampledsv := "../../testdata/iris/iris.dsv"
	// oob file output
	oobFile := "oobiris"

	runRandomForest(t, sampledsv, NTREE, NBOOTSTRAP, FEATSTART, FEATEND,
		oobFile)
}

func TestEnsemblingPhoneme(t *testing.T) {
	// input data
	sampledsv := "../../testdata/phoneme/phoneme.dsv"
	// oob file output
	oobFile := "oobphoneme"

	FEATSTART = 3
	FEATEND = 4

	runRandomForest(t, sampledsv, NTREE, NBOOTSTRAP, FEATSTART, FEATEND,
		oobFile)
}

func TestEnsemblingPhonemeSMOTE(t *testing.T) {
	// input data
	sampledsv := "../../resampling/smote/phoneme_smote.dsv"
	// oob file output
	oobFile := "oobphonemesmote"

	FEATSTART = 3
	FEATEND = 4

	runRandomForest(t, sampledsv, NTREE, NBOOTSTRAP, FEATSTART, FEATEND,
		oobFile)
}

func TestEnsemblingPhonemeLNSMOTE(t *testing.T) {
	// input data
	sampledsv := "../../resampling/lnsmote/phoneme_lnsmote.dsv"
	// oob file output
	oobFile := "oobphonemelnsmote"

	FEATSTART = 3
	FEATEND = 4

	runRandomForest(t, sampledsv, NTREE, NBOOTSTRAP, FEATSTART, FEATEND,
		oobFile)
}
