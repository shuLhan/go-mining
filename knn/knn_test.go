// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package knn_test

import (
	"fmt"
	"testing"

	"github.com/shuLhan/dsv"
	"github.com/shuLhan/dsv/util/assert"
	"github.com/shuLhan/go-mining/knn"
)

func TestComputeEuclidianDistance(t *testing.T) {
	var exp = []string{
		`[0.302891 0.608544 0.47413 1.42718 -0.811085 1]`,
		`[[0.1048 0.5756 0.3424 0.8304 0 1] 1.3306502169991932
 [0.318095 0.810884 0.818231 0.820952 0.860136 1] 1.684961127148042
 [0.540984 2.1451 -0.561563 -0.227764 -0.140565 1] 2.2662316739468626
 [1.03941 2.27108 1.69893 -0.36917 -0.167506 1] 2.4624751775398668
 [0.655083 1.0539 1.37163 -0.723877 1.77021 1] 2.535231744831229
]`,
	}

	// Reading data
	reader := dsv.NewReader()
	e := dsv.Open(reader, "../testdata/phoneme/phoneme.dsv")

	if nil != e {
		return
	}

	dsv.Read(reader)
	reader.Close()

	// Processing
	input := knn.Input{
		DistanceMethod: knn.TEuclidianDistance,
		ClassIdx:       5,
		K:              5,
	}

	classes := reader.Rows.GroupByValue(input.K)

	_, input.Dataset = classes.GetMinority()

	kneighbors, e := input.Neighbors(0)
	got := fmt.Sprint(kneighbors)

	assert.Equal(t, exp[1], got)
}
