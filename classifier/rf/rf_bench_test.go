// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rf_test

import (
	"testing"
)

func BenchmarkPhoneme(b *testing.B) {
	SampleDsvFile = "../../testdata/phoneme/phoneme.dsv"
	OobFile = "phoneme.oob"

	MinFeature = 3
	MaxFeature = 4

	for x := 0; x < b.N; x++ {
		runRandomForest()
	}
}
