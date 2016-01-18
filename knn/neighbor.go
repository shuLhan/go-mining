// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package knn

import (
	"github.com/shuLhan/dsv"
)

/*
Neighbors is a mapping between sample and their distance.
This type implement the sort interface.
*/
type Neighbors struct {
	// Dataset contain the data in neighbors
	dsv.Dataset
	// Distance value
	Distances []float64
}

/*
Len return the number of neighbors.
This is for sort interface.
*/
func (neighbors *Neighbors) Len() int {
	return len(neighbors.Distances)
}

/*
Less return true if i < j.
This is for sort interface.
*/
func (neighbors *Neighbors) Less(i, j int) bool {
	if neighbors.Distances[i] < neighbors.Distances[j] {
		return true
	}
	return false
}

/*
Swap content of object in index i with index j.
This is for sort interface.
*/
func (neighbors *Neighbors) Swap(i, j int) {
	row := neighbors.Rows[i]
	distance := neighbors.Distances[i]

	neighbors.Rows[i] = neighbors.Rows[j]
	neighbors.Distances[i] = neighbors.Distances[j]

	neighbors.Rows[j] = row
	neighbors.Distances[j] = distance
}

/*
Add new neighbor.
*/
func (neighbors *Neighbors) Add(row dsv.Row, distance float64) {
	neighbors.PushRow(row)
	neighbors.Distances = append(neighbors.Distances, distance)
}

/*
SelectRange select all neighbors from index `start` to `end`.
Return an empty set if start or end is out of range.
*/
func (neighbors *Neighbors) SelectRange(start, end int) (newn Neighbors) {
	if start < 0 {
		return
	}

	if end > neighbors.Len() {
		return
	}

	for x := start; x < end; x++ {
		row := neighbors.GetRow(x)
		newn.Add(*row, neighbors.Distances[x])
	}
	return
}
