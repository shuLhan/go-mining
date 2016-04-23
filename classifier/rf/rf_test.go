// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rf_test

import (
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/classifier/rf"
	"github.com/shuLhan/tabula"
	"log"
	"testing"
)

// Global options to run for each test.
var (
	// SampleDsvFile is the file that contain samples config.
	SampleDsvFile string
	// DoTest if its true then the dataset will splited into training and
	// test set with random selection without replacement.
	DoTest = false
	// NTree number of tree to generate.
	NTree = 100
	// NBootstrap percentage of sample used as subsample.
	NBootstrap = 66
	// MinFeature number of feature to begin with.
	MinFeature = 1
	// MaxFeature maximum number of feature to test
	MaxFeature = -1
	// RunOOB if its true the the OOB samples will be used to test the
	// model in each iteration.
	RunOOB = true
	// OobFile is the file where OOB statistic will be saved.
	OobFile string
)

func getSamples() (train, test tabula.ClasetInterface) {
	samples := tabula.Claset{}
	_, e := dsv.SimpleRead(SampleDsvFile, &samples)
	if nil != e {
		log.Fatal(e)
	}

	if !DoTest {
		return &samples, nil
	}

	ntrain := int(float32(samples.Len()) * (float32(NBootstrap) / 100.0))

	bag, oob, _, _ := tabula.RandomPickRows(&samples, ntrain, false)

	train = bag.(tabula.ClasetInterface)
	test = oob.(tabula.ClasetInterface)

	train.SetClassIndex(samples.GetClassIndex())
	test.SetClassIndex(samples.GetClassIndex())

	return train, test
}

//
// writeOOB will save the oob data to file.
//
func writeOOB(dataOOB *tabula.Dataset) error {
	writer, e := dsv.NewWriter("")
	if e != nil {
		return e
	}

	e = writer.OpenOutput(OobFile)
	if e != nil {
		return e
	}

	_, e = writer.WriteRawDataset(dataOOB, nil)
	if e != nil {
		return e
	}

	return writer.Close()
}

func runRandomForest() {
	trainset, testset := getSamples()

	// dataset to save each oob error in each feature iteration.
	dataooberr := tabula.NewDataset(tabula.DatasetModeColumns, nil, nil)

	if MaxFeature < 0 {
		MaxFeature = trainset.GetNColumn()
	}

	for nfeature := MinFeature; nfeature < MaxFeature; nfeature++ {
		// Create and build random forest.
		forest := rf.New(NTree, nfeature, NBootstrap, RunOOB)

		e := forest.Build(trainset)

		if e != nil {
			log.Fatal(e)
		}

		if RunOOB {
			// Save OOB error based on number of feature.
			colName := fmt.Sprintf("M%d", nfeature)

			col := tabula.NewColumnReal(forest.Stats().
				OobErrorMeans(), colName)

			dataooberr.PushColumn(*col)
		}
		if DoTest {
			predicts, _, probs := forest.ClassifySet(testset, nil)

			perfs := forest.Performance(testset, predicts, probs)

			e := perfs.Write("phoneme.perfs")
			if e != nil {
				log.Fatal(e)
			}
		}
	}

	if RunOOB {
		e := writeOOB(dataooberr)
		if e != nil {
			log.Fatal(e)
		}
	}
}

func TestEnsemblingGlass(t *testing.T) {
	SampleDsvFile = "../../testdata/forensic_glass/fgl.dsv"
	OobFile = "glass.oob"

	runRandomForest()
}

func TestEnsemblingIris(t *testing.T) {
	SampleDsvFile = "../../testdata/iris/iris.dsv"
	OobFile = "iris.oob"

	runRandomForest()
}

func TestEnsemblingPhoneme(t *testing.T) {
	SampleDsvFile = "../../testdata/phoneme/phoneme.dsv"
	OobFile = "phoneme.oob"

	NTree = 50
	MinFeature = 3
	MaxFeature = 4
	RunOOB = false
	DoTest = true

	runRandomForest()
}

func TestEnsemblingSmotePhoneme(t *testing.T) {
	SampleDsvFile = "../../resampling/smote/phoneme_smote.dsv"
	OobFile = "phonemesmote.oob"

	MinFeature = 3
	MaxFeature = 4

	runRandomForest()
}

func TestEnsemblingLnsmotePhoneme(t *testing.T) {
	SampleDsvFile = "../../resampling/lnsmote/phoneme_lnsmote.dsv"
	OobFile = "phonemelnsmote.oob"

	MinFeature = 3
	MaxFeature = 4

	runRandomForest()
}

func TestWvc2010Lnsmote(t *testing.T) {
	SampleDsvFile = "../../testdata/wvc2010lnsmote/wvc2010_features.lnsmote.dsv"
	OobFile = "wvc2010lnsmote.oob"

	NTree = 1
	MinFeature = 5
	MaxFeature = 6

	runRandomForest()
}
