// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package knn

import (
	"github.com/shuLhan/dsv"
)

/*
Neighbor is a mapping between sample and their distance.
This type implement the sort interface.
*/
type Neighbor struct {
	// Sample of data
	Sample dsv.Row
	// Distance value
	Distance float64
}

/*
Neighbors is a slice of Neighbor
*/
type Neighbors []Neighbor

/*
Len return length of slice.
*/
func (jarak *Neighbors) Len() int {
	return len(*jarak)
}

/*
Less return true if i < j.
*/
func (jarak *Neighbors) Less(i, j int) bool {
	if (*jarak)[i].Distance < (*jarak)[j].Distance {
		return true
	}
	return false
}

/*
Swap content of object in index i with index j.
*/
func (jarak *Neighbors) Swap(i, j int) {
	var tmp = &Neighbor{}

	tmp.Sample = (*jarak)[i].Sample
	tmp.Distance = (*jarak)[i].Distance

	(*jarak)[i].Sample = (*jarak)[j].Sample
	(*jarak)[i].Distance = (*jarak)[j].Distance

	(*jarak)[j].Sample = tmp.Sample
	(*jarak)[j].Distance = tmp.Distance
}
