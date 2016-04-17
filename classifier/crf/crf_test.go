// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crf_test

import (
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/classifier/crf"
	"github.com/shuLhan/tabula"
	"testing"
)

func TestCascadedRF(t *testing.T) {
	sampledsv := "../../testdata/phoneme/phoneme.dsv"

	// read trainingset.
	samples := tabula.Claset{}
	_, e := dsv.SimpleRead(sampledsv, &samples)
	if e != nil {
		t.Fatal(e)
	}

	nbag := (samples.Len() * 63) / 100
	train, test, _, testIds := tabula.RandomPickRows(&samples, nbag, false)

	trainset := train.(tabula.ClasetInterface)
	testset := test.(tabula.ClasetInterface)

	crf := crf.Runtime{}

	e = crf.Build(trainset)
	if e != nil {
		t.Fatal(e)
	}

	testset.RecountMajorMinor()
	fmt.Println("Testset:", testset)

	_, cm := crf.ClassifySetByWeight(testset, testIds)

	fmt.Println("Confusion matrix:", cm)
}
