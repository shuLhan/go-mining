// Copyright 2015-2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cascadedrf

import (
	"github.com/shuLhan/go-mining/classifiers/randomforest"
)

//
// Stage define the stage for cascaded process.
// Each stage contain random-forest runtime and weight of forest.
//
type Stage struct {
	// RandomForest contain all tree in the forest.
	RandomForest randomforest.Runtime
	// Weight
	Weight float64
}
