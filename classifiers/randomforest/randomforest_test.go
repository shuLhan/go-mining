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

func TestEnsembling(t *testing.T) {
	// read data.
	samples, e := dataset.NewReader("../../testdata/forensic_glass/glass.dsv")

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

	// number of tree to generate.
	ntree := 500
	// percentage of sample used as subsample.
	npercent := 63

	nfeature := samples.GetNColumn()
	n := 2
	//nfeature := n + 1

	for ; n < nfeature; n++ {
		// generate random forest.
		forest, oobsteps, e := randomforest.Ensembling(samples, ntree,
			n, npercent)

		if e != nil {
			t.Fatal(e)
		}

		colName := fmt.Sprintf("M%d", n)

		col := dsv.NewColumnReal(oobsteps, colName)

		dataooberr.PushColumn(*col)

		glog.V(2).Info(forest)
	}

	// write oob data to file
	writer, e := dsv.NewWriter("")

	if e != nil {
		t.Fatal(e)
	}

	e = writer.OpenOutput("oob")

	if e != nil {
		t.Fatal(e)
	}

	_, e = writer.WriteDataset(dataooberr, nil)
	if e != nil {
		t.Fatal(e)
	}
}
