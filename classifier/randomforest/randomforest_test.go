// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package randomforest_test

import (
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/classifier/randomforest"
	"github.com/shuLhan/tabula"
	"log"
	"testing"
)

var (
	// NTREE number of tree to generate.
	NTREE = 100
	// NBOOTSTRAP percentage of sample used as subsample.
	NBOOTSTRAP = 66
	// FEATSTART number of feature to begin with.
	FEATSTART = 1
	// FEATEND maximum number of feature to test
	FEATEND = -1
)

func runRandomForest(sampledsv string,
	ntree, npercent, nfeature, maxFeature int,
	oobFile string,
) {
	// read data.
	samples := tabula.Claset{}
	_, e := dsv.SimpleRead(sampledsv, &samples)
	if nil != e {
		log.Fatal(e)
	}

	// dataset to save each oob error in each feature iteration.
	dataooberr := tabula.NewDataset(tabula.DatasetModeColumns, nil, nil)

	if maxFeature < 0 {
		maxFeature = samples.GetNColumn()
	}

	for ; nfeature < maxFeature; nfeature++ {
		// Create and build random forest.
		forest := randomforest.New(ntree, nfeature, npercent)

		e := forest.Build(&samples)

		if e != nil {
			log.Fatal(e)
		}

		// Save OOB error based on number of feature.
		colName := fmt.Sprintf("M%d", nfeature)

		col := tabula.NewColumnReal(forest.Stats().OobErrorMeans(),
			colName)

		dataooberr.PushColumn(*col)
	}

	// write oob data to file
	writer, e := dsv.NewWriter("")

	if e != nil {
		log.Fatal(e)
	}

	e = writer.OpenOutput(oobFile)

	if e != nil {
		log.Fatal(e)
	}

	_, e = writer.WriteRawDataset(dataooberr, nil)
	if e != nil {
		log.Fatal(e)
	}
}

func TestEnsemblingGlass(t *testing.T) {
	sampledsv := "../../testdata/forensic_glass/fgl.dsv"
	// oob file output
	oobFile := "glass.oob"

	runRandomForest(sampledsv, NTREE, NBOOTSTRAP, FEATSTART, FEATEND,
		oobFile)
}

func TestEnsemblingIris(t *testing.T) {
	// input data
	sampledsv := "../../testdata/iris/iris.dsv"
	// oob file output
	oobFile := "iris.oob"

	runRandomForest(sampledsv, NTREE, NBOOTSTRAP, FEATSTART, FEATEND,
		oobFile)
}

func TestEnsemblingPhoneme(t *testing.T) {
	// input data
	sampledsv := "../../testdata/phoneme/phoneme.dsv"
	// oob file output
	oobFile := "phoneme.oob"

	FEATSTART = 3
	FEATEND = 4

	runRandomForest(sampledsv, NTREE, NBOOTSTRAP, FEATSTART, FEATEND,
		oobFile)
}

func TestEnsemblingSmotePhoneme(t *testing.T) {
	// input data
	sampledsv := "../../resampling/smote/phoneme_smote.dsv"
	// oob file output
	oobFile := "phonemesmote.oob"

	FEATSTART = 3
	FEATEND = 4

	runRandomForest(sampledsv, NTREE, NBOOTSTRAP, FEATSTART, FEATEND,
		oobFile)
}

func TestEnsemblingLnsmotePhoneme(t *testing.T) {
	// input data
	sampledsv := "../../resampling/lnsmote/phoneme_lnsmote.dsv"
	// oob file output
	oobFile := "phonemelnsmote.oob"

	FEATSTART = 3
	FEATEND = 4

	runRandomForest(sampledsv, NTREE, NBOOTSTRAP, FEATSTART, FEATEND,
		oobFile)
}

func TestWvc2010Lnsmote(t *testing.T) {
	sampledsv := "../../testdata/wvc2010lnsmote/wvc2010_features.lnsmote.dsv"
	oobFile := "wvc2010lnsmote.oob"

	NTREE = 1
	FEATSTART = 5
	FEATEND = 6

	runRandomForest(sampledsv, NTREE, NBOOTSTRAP, FEATSTART, FEATEND,
		oobFile)
}

func BenchmarkPhoneme(b *testing.B) {
	// input data
	sampledsv := "../../testdata/phoneme/phoneme.dsv"
	// oob file output
	oobFile := "phoneme.oob"

	FEATSTART = 3
	FEATEND = 4

	for x := 0; x < b.N; x++ {
		runRandomForest(sampledsv, NTREE, NBOOTSTRAP, FEATSTART,
			FEATEND, oobFile)
	}
}
