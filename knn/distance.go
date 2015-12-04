// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package knn

import (
	"fmt"
	"strconv"

	"github.com/shuLhan/dsv"
)

/*
Distance is a mapping between record and their distance.
*/
type Distance struct {
	Sample	dsv.Row
	Value	float64
}

/*
DistanceSlice is a slice of Distance
*/
type DistanceSlice []Distance

/*
NewDistance create new distance object.
*/
func NewDistance() Distance {
	return Distance {
		Sample	: nil,
		Value	: 0.0,
	}
}

/*
String format the distance object.
*/
func (d Distance) String () (s string) {
	s += fmt.Sprint (d.Sample)
	s += " "
	s += strconv.FormatFloat (d.Value, 'f', -1, 64)
	s += "\n"

	return s
}

/*
Len return length of slice.
*/
func (jarak *DistanceSlice) Len () int {
	return len (*jarak)
}

/*
Less return true if i < j.
*/
func (jarak *DistanceSlice) Less (i, j int) bool {
	if (*jarak)[i].Value < (*jarak)[j].Value {
		return true
	}
	return false
}

/*
Swap content of object in index i with index j.
*/
func (jarak *DistanceSlice) Swap (i, j int) {
	var tmp = &Distance{};

	tmp.Sample = (*jarak)[i].Sample
	tmp.Value = (*jarak)[i].Value

	(*jarak)[i].Sample = (*jarak)[j].Sample
	(*jarak)[i].Value = (*jarak)[j].Value

	(*jarak)[j].Sample = tmp.Sample
	(*jarak)[j].Value = tmp.Value
}
