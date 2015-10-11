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
	Record	*dsv.RecordSlice
	Value	float64
}

/*
DistanceSlice is a slice of Distance
*/
type DistanceSlice []Distance

/*
NewDistance create new distance object.
*/
func NewDistance () *Distance {
	return &Distance {
		Record	: nil,
		Value	: 0.0,
	}
}

/*
String format the distance object.
*/
func (d Distance) String () (s string) {
	s += fmt.Sprint (d.Record)
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

	tmp.Record = (*jarak)[i].Record
	tmp.Value = (*jarak)[i].Value

	(*jarak)[i].Record = (*jarak)[j].Record
	(*jarak)[i].Value = (*jarak)[j].Value

	(*jarak)[j].Record = tmp.Record
	(*jarak)[j].Value = tmp.Value
}
