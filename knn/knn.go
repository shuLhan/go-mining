// Copyright 2015-2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package knn implement the K Nearest Neighbor using Euclidian to compute the
distance between samples.
*/
package knn

import (
	"fmt"
	"github.com/shuLhan/tabula"
	"math"
	"os"
	"sort"
	"strconv"
)

const (
	// TEuclidianDistance used in Runtime.DistanceMethod.
	TEuclidianDistance = 0
)

var (
	// DEBUG debug level for this package, set from environment.
	DEBUG = 0
)

//
// Runtime parameters for KNN processing.
//
type Runtime struct {
	// DistanceMethod define how the distance between sample will be
	// measured.
	DistanceMethod int
	// ClassIndex define index of class in dataset.
	ClassIndex int `json:"ClassIndex"`
	// K define number of nearest neighbors that will be searched.
	K int `json:"K"`

	// AllNeighbors contain all neighbours
	AllNeighbors Neighbors
}

func init() {
	var e error
	DEBUG, e = strconv.Atoi(os.Getenv("KNN_DEBUG"))
	if e != nil {
		DEBUG = 0
	}
}

/*
ComputeEuclidianDistance compute the distance of instance with each sample in
dataset `samples` and return it.
*/
func (in *Runtime) ComputeEuclidianDistance(samples *tabula.Rows,
	instance *tabula.Row,
) {
	for x := range *samples {
		row := (*samples)[x]

		// compute euclidian distance
		d := 0.0
		for y, rec := range *row {
			if y == in.ClassIndex {
				// skip class attribute
				continue
			}

			ir := (*instance)[y]
			diff := 0.0

			diff = ir.Float() - rec.Float()

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
a slice of neighbors.
*/
func (in *Runtime) FindNeighbors(samples *tabula.Rows, instance *tabula.Row) (
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

	if DEBUG >= 2 {
		fmt.Println("[knn] all neighbors:", in.AllNeighbors.Len())
	}

	kneighbors = in.AllNeighbors.SelectRange(0, minK)

	if DEBUG >= 2 {
		fmt.Println("[knn] k neighbors:", kneighbors.Len())
	}

	return
}
