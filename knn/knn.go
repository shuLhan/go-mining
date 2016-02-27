// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package knn implement the K Nearest Neighbor using Euclidian to compute the
distance between samples.
*/
package knn

import (
	"github.com/golang/glog"
	"github.com/shuLhan/tabula"
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
	// AllNeighbors contain all neighbours
	AllNeighbors Neighbors
}

/*
ComputeEuclidianDistance compute the distance of instance with each sample in
dataset `samples` and return it.
*/
func (in *Input) ComputeEuclidianDistance(samples tabula.Rows,
	instance tabula.Row) {
	for _, row := range samples {
		// compute euclidian distance
		d := 0.0
		for y, rec := range row {
			if y == in.ClassIdx {
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

		// only add sample distance which is not zero (its probably
		// we calculating with the instance itself)
		if d != 0 {
			in.AllNeighbors.Add(row, math.Sqrt(d))
		}
	}

	sort.Sort(&in.AllNeighbors)
}

/*
FindNeighbors Given sample set and an instance, return the nearest neighbors as
a slice of neighbours.
*/
func (in *Input) FindNeighbors(samples tabula.Rows, instance tabula.Row) (
	kneighbors Neighbors,
) {
	// Reset current input neighbours
	in.AllNeighbors = Neighbors{}

	switch in.DistanceMethod {
	case TEuclidianDistance:
		in.ComputeEuclidianDistance(samples, instance)
	}

	// Make sure number of neighbors is greater than request.
	minK := in.AllNeighbors.Len()
	if minK > in.K {
		minK = in.K
	}

	glog.V(2).Info(">>> all neighbors:", in.AllNeighbors.Len())

	kneighbors = in.AllNeighbors.SelectRange(0, minK)

	glog.V(2).Info(">>> k neighbors:", kneighbors.Len())

	return
}
