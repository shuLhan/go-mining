// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crf_test

import (
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/classifier"
	"github.com/shuLhan/go-mining/classifier/crf"
	"github.com/shuLhan/tabula"
	"testing"
)

var (
	SampleFile string
	PerfFile   string
	StatFile   string
	NStage     = 200
	NTree      = 1
)

func runCRF(t *testing.T) {
	// read trainingset.
	samples := tabula.Claset{}
	_, e := dsv.SimpleRead(SampleFile, &samples)
	if e != nil {
		t.Fatal(e)
	}

	nbag := (samples.Len() * 63) / 100
	train, test, _, testIds := tabula.RandomPickRows(&samples, nbag, false)

	trainset := train.(tabula.ClasetInterface)
	testset := test.(tabula.ClasetInterface)

	crf := crf.Runtime{
		Runtime: classifier.Runtime{
			StatFile: StatFile,
			PerfFile: PerfFile,
		},
		NStage: NStage,
		NTree:  NTree,
	}

	e = crf.Build(trainset)
	if e != nil {
		t.Fatal(e)
	}

	testset.RecountMajorMinor()
	fmt.Println("Testset:", testset)

	predicts, cm, probs := crf.ClassifySetByWeight(testset, testIds)

	fmt.Println("Confusion matrix:", cm)

	crf.Performance(testset, predicts, probs)
	e = crf.WritePerformance()
	if e != nil {
		t.Fatal(e)
	}
}

func TestPhoneme200_1(t *testing.T) {
	SampleFile = "../../testdata/phoneme/phoneme.dsv"
	PerfFile = "phoneme_200_1.perf"
	StatFile = "phoneme_200_1.stat"

	runCRF(t)
}

func TestPhoneme200_10(t *testing.T) {
	SampleFile = "../../testdata/phoneme/phoneme.dsv"
	PerfFile = "phoneme_200_10.perf"
	StatFile = "phoneme_200_10.stat"
	NTree = 10

	runCRF(t)
}
