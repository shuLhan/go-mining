// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package randomforest_test

import (
	"github.com/golang/glog"
	"github.com/shuLhan/go-mining/dataset"
	"github.com/shuLhan/go-mining/classifiers/randomforest"
	"io"
	"math"
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

	// number of tree to generate.
	ntree := 100
	// number of feature selected randomly.
	n := len(samples.InputMetadata) - 1
	nfeature := int(math.Sqrt(float64(n)))
	// percentage of sample used as subsample.
	npercent := 66

	// generate random forest.
	forest, e := randomforest.Ensembling(samples, ntree, nfeature,
						npercent)

	if e != nil {
		t.Fatal(e)
	}

	glog.V(2).Info(forest)
}
