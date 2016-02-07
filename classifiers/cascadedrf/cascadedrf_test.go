// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cascadedrf_test

import (
	"github.com/shuLhan/go-mining/classifiers/cascadedrf"
	"github.com/shuLhan/go-mining/dataset"
	"testing"
)

func TestCascadedRF(t *testing.T) {
	nstage := 10
	ntree := 1
	percentboot := 66
	nfeature := 3
	tprate := 0.6
	tnrate := 0.6
	sampledsv := "../../testdata/iris/iris.dsv"

	// read samples.
	samples, e := dataset.NewReader(sampledsv)

	if nil != e {
		t.Fatal(e)
	}

	crf := cascadedrf.New(nstage, ntree, percentboot, nfeature, tprate, tnrate)

	crf.Train(samples)
}
