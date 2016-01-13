// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package knn implement the K Nearest Neighbor using Euclidian to compute the
distance between samples.
*/
package knn

import (
	"github.com/shuLhan/dsv"
	"math"
	"sort"
)

const (
	// TEuclidianDistance used in Input.DistanceMethod.
	TEuclidianDistance = 0
)

/*
Input parameters for KNN processing.
*/
type Input struct {
	// DistanceMethod define how the distance between sample will be
	// measured.
	DistanceMethod int
	// ClassIdx define index of class in dataset.
	ClassIdx int
	// K define number of nearset neighbors that will be searched.
	K int
}

/*
ComputeEuclidianDistance compute the distance of sample in index
`instanceIdx` with each sample in dataset `samples` and return it.
*/
func (input *Input) ComputeEuclidianDistance(samples dsv.Rows, instanceIdx int) (
	neighbors Neighbors,
) {
	instance := samples[instanceIdx]

	for x, row := range samples {
		if instanceIdx == x {
			continue
		}

		// compute euclidian distance
		d := 0.0
		for y, rec := range row {
			if y == input.ClassIdx {
				// skip class attribute
				continue
			}

			ir := instance[y]
			diff := 0.0

			switch ir.V.(type) {
			case float64:
				diff = ir.Value().(float64) -
					rec.Value().(float64)
			case int64:
				diff = float64(ir.Value().(int64) -
					rec.Value().(int64))
			}

			d += math.Abs(diff)
		}

		neighbors = append(neighbors, Neighbor{row, math.Sqrt(d)})
	}

	sort.Sort(&neighbors)

	return
}

/*
FindNeighbors Given sample set and instance index, return the nearest neighbors as
a slice of distance.
*/
func (input *Input) FindNeighbors(samples dsv.Rows, instanceIdx int) (
	kneighbors Neighbors,
) {
	switch input.DistanceMethod {
	case TEuclidianDistance:
		kneighbors = input.ComputeEuclidianDistance(samples,
			instanceIdx)
	}

	// Make sure number of neighbors is greater than request.
	minK := len(kneighbors)
	if minK > input.K {
		minK = input.K
	}

	kneighbors = kneighbors[0:minK]

	return
}
