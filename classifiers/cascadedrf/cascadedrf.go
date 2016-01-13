// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package cascadedrf implement the cascaded random forest algorithm, proposed by
Baumann et.al in their paper:

	Baumann, Florian, et al. "Cascaded Random Forest for Fast Object
	Detection." Image Analysis. Springer Berlin Heidelberg, 2013. 131-142.

*/
package cascadedrf

import (
	"github.com/shuLhan/go-mining/classifiers/randomforest"
	"github.com/shuLhan/go-mining/dataset"
)

/*
Input
*/
type Input struct {
	// NStage number of stages
	NStage int
	// NTree number of tree in each stage
	NTree int
	// PercentBoot percentage of bootstrap
	PercentBoot int
	// NFeature number of features used to split the dataset.
	NFeature int
	// Stages contain list of cascaded stages.
	Stages []Stage
	// TPR true positive rate per stage
	TPR float64
	// TNR true negative rate per stage
	TNR float64
}

/*
New create and return new input for cascaded random-forest.
*/
func New(nstage, ntree, percentboot, nfeature int, tprate, tnrate float64) (
	crf *Input,
) {
	crf = &Input{
		NStage:      nstage,
		NTree:       ntree,
		PercentBoot: percentboot,
		NFeature:    nfeature,
		TPR:         tprate,
		TNR:         tnrate,
	}
	return
}

/*
Train given a sample dataset, build the stage and random forest.
*/
func (crf *Input) Train(samples *dataset.Reader) {
	for x := 0; x < crf.NStage; x++ {
		forest := randomforest.New(crf.NTree, crf.NFeature,
			crf.PercentBoot)

		forest.Init(samples)

		totalOobErr := 0.0

		for t := 0; t < crf.NTree; t++ {
			oobErr, e = forest.GrownTree(t, samples, totalOOBErr)
			totalOOBErr += oobErr
			tprate := 1 - (float64(totalOOBErr) / float64(t+1))

			if tprate > crf.TPR {
				break
			}
		}
	}
}
