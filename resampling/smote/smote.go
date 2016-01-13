// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package smote resamples a dataset by applying the Synthetic Minority
Oversampling TEchnique (SMOTE). The original dataset must fit entirely in
memory.  The amount of SMOTE and number of nearest neighbors may be specified.
For more information, see

	Nitesh V. Chawla et. al. (2002). Synthetic Minority Over-sampling
	Technique. Journal of Artificial Intelligence Research. 16:321-357.
*/
package smote

import (
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/knn"
	"math/rand"
	"time"
)

/*
SMOTE parameters for input and output.
*/
type SMOTE struct {
	// Input the K-Nearest-Neighbourhood parameters.
	knn.Input
	// PercentOver input for oversampling percentage.
	PercentOver int
	// n input for number of new synthetic per sample.
	n int
	// Synthetic output for new sample.
	Synthetic dsv.Rows
}

const (
	// DefaultK nearest neighbors.
	DefaultK = 5
	// DefaultPercentOver sampling.
	DefaultPercentOver = 100
)

/*
Init parameter, set to default value if it not valid.
*/
func (smote *SMOTE) Init() {
	rand.Seed(time.Now().UnixNano())

	if smote.K <= 0 {
		smote.K = DefaultK
	}
	if smote.PercentOver <= 0 {
		smote.PercentOver = DefaultPercentOver
	}
}

/*
populate will generate new synthetic sample using nearest neighbors.
*/
func (smote *SMOTE) populate(instance *dsv.Row, neighbors knn.Neighbors) {
	var i, n, lenAttr, attr int
	var iv, sv, dif, gap, newAttr float64
	var sample dsv.Row
	var ir *dsv.Record
	var sr *dsv.Record

	lenAttr = len(*instance)

	for i = 0; i < smote.n; i++ {
		// choose one of the K nearest neighbors
		n = rand.Intn(len(neighbors))
		sample = neighbors[n].Sample

		newSynt := make(dsv.Row, lenAttr)

		// Compute new synthetic attributes.
		for attr = range sample {
			if attr == smote.ClassIdx {
				continue
			}

			ir = (*instance)[attr]
			sr = sample[attr]

			iv = ir.Value().(float64)
			sv = sr.Value().(float64)

			dif = sv - iv
			gap = rand.Float64()
			newAttr = iv + (gap * dif)

			record := &dsv.Record{}
			record.SetFloat(newAttr)
			newSynt[attr] = record
		}

		newSynt[smote.ClassIdx] = (*instance)[attr]

		smote.Synthetic.PushBack(newSynt)
	}
}

/*
Resampling will run SMOTE algorithm using parameters that has been defined in
struct and return list of synthetic samples.
*/
func (smote *SMOTE) Resampling(dataset dsv.Rows) dsv.Rows {
	smote.Init()

	if smote.PercentOver < 100 {
		// Randomize minority class sample by percentage.
		smote.n = (smote.PercentOver / 100.0) * len(dataset)
		dataset, _, _, _ = dataset.RandomPick(smote.n, false)
		smote.PercentOver = 100
	}
	smote.n = smote.PercentOver / 100.0

	// for each sample in dataset, generate their synthetic samples.
	for i := range dataset {
		// Compute k nearest neighbors for instance
		neighbors := smote.FindNeighbors(dataset, i)

		smote.populate(&dataset[i], neighbors)
	}

	return smote.Synthetic
}
