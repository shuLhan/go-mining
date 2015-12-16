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
		`[[0.243474 0.505146 0.472892 1.34802 -0.844252 1] 0.5257185558832786
 [0.202343 0.485983 0.527533 1.47307 -0.809672 1] 0.5690474496911485
 [0.215496 0.523418 0.51719 1.43548 -0.933981 1] 0.5888777462258191
 [0.214331 0.546086 0.414773 1.38542 -0.702336 1] 0.6007362149895741
 [0.301676 0.554505 0.594757 1.21258 -0.873084 1] 0.672666336306493
]`,
	}

	// Reading data
	reader, e := dsv.NewReader("../testdata/phoneme/phoneme.dsv")

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
