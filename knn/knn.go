// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package knn

import (
	"math"
	"sort"

	"github.com/shuLhan/dsv"
)

const (
	// TEuclidianDistance used in Input.DistanceMethod.
	TEuclidianDistance = 0
)

/*
Input parameters for KNN processing.
*/
type Input struct {
	// Dataset training data.
	Dataset		dsv.Rows
	// DistanceMethod define how the distance between sample will be
	// measured.
	DistanceMethod	int
	// ClassIdx define index of class in dataset.
	ClassIdx	int
	// K define number of nearset neighbors that will be searched.
	K		int
}

/*
ComputeEuclidianDistance of instance with each sample in dataset.
*/
func (input *Input) ComputeEuclidianDistance (instanceIdx int) (
				distances DistanceSlice, e error) {
	instance := input.Dataset[instanceIdx]

	for rowIdx, row := range input.Dataset {
		if instanceIdx == rowIdx {
			continue
		}

		// compute euclidian distance
		d := 0.0
		for i := range row {
			if i == input.ClassIdx {
				// skip class attribute
				continue
			}

			ir := instance[i]
			sr := row[i]

			switch ir.V.(type) {
			case float64:
				d += math.Abs (ir.Value ().(float64) - sr.Value ().(float64))
			case int64:
				d += math.Abs (float64(ir.Value ().(int64) - sr.Value ().(int64)))
			}
		}

		distances = append(distances, Distance{row, math.Sqrt(d)})
	}

	sort.Sort (&distances)

	return
}

/*
Neighbors return the nearest neighbors as a slice of distance.
*/
func (input *Input) Neighbors(instanceIdx int) (kneighbors DistanceSlice,
						e error) {
	switch (input.DistanceMethod) {
	case TEuclidianDistance:
		kneighbors, e = input.ComputeEuclidianDistance(instanceIdx)
	}

	if nil != e {
		return nil, e
	}

	kneighbors = kneighbors[0:input.K]

	return
}
