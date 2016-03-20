// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cascadedrf

import (
	"github.com/shuLhan/go-mining/classifiers/randomforest"
)

type Stage struct {
	// RandomForest contain all tree in the forest.
	RandomForest []randomforest.Runtime
	// Weight
	Weight float64
}
