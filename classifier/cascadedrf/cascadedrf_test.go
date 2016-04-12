// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cascadedrf_test

import (
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/classifier/cascadedrf"
	"github.com/shuLhan/tabula"
	"testing"
)

func TestCascadedRF(t *testing.T) {
	sampledsv := "../../testdata/phoneme/phoneme.dsv"

	// read samples.
	samples := tabula.Claset{}
	_, e := dsv.SimpleRead(sampledsv, &samples)
	if e != nil {
		t.Fatal(e)
	}

	crf := cascadedrf.Runtime{}

	e = crf.Build(&samples)
	if e != nil {
		t.Fatal(e)
	}
}
