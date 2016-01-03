// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package randomforest_test

import (
	"fmt"
	"github.com/golang/glog"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/classifiers/randomforest"
	"github.com/shuLhan/go-mining/dataset"
	"io"
	"testing"
)

const (
	// NTREE number of tree to generate.
	NTREE = 1
	// NBOOTSTRAP percentage of sample used as subsample.
	NBOOTSTRAP = 63
	// FEATSTART number of feature to begin with.
	FEATSTART = 3
	// FEATEND maximum number of feature to test
	FEATEND = 4
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
	dataooberr, e := dsv.NewDataset(dsv.DatasetModeColumns, nil, nil)

	if e != nil {
		t.Fatal(e)
	}

	if nEnd < 0 {
		nEnd = samples.GetNColumn()
	}

	for ; nStart < nEnd; nStart++ {
		// generate random forest.
		forest, oobsteps, e := randomforest.Ensembling(samples, ntree,
			nStart, npercent)

		if e != nil {
			t.Fatal(e)
		}

		colName := fmt.Sprintf("M%d", nStart)

		col := dsv.NewColumnReal(oobsteps, colName)

		e = dataooberr.PushColumn(*col)
		if e != nil {
			t.Fatal(e)
		}

		glog.V(2).Info(forest)
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

	_, e = writer.WriteDataset(dataooberr, nil)
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
