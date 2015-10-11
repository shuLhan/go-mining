package knn

import (
	"container/list"
	"math"
	"sort"

	"github.com/shuLhan/dsv"
)

const (
	// TEuclidianDistance used in Input.Method.
	TEuclidianDistance = 0
)

/*
Input parameters for KNN processing.
*/
type Input struct {
	Dataset		*dsv.Row
	Method		int
	ClassIdx	int
	K		int
}

/*
ComputeEuclidianDistance of instance with each sample in dataset.
*/
func (input *Input) ComputeEuclidianDistance (instance *dsv.RecordSlice) (
				*DistanceSlice, error) {
	var e error
	var i int
	var d float64
	var sv *dsv.RecordSlice
	var ir *dsv.Record
	var sr *dsv.Record
	var sample *list.Element

	distances := make (DistanceSlice, 0)

	for sample = input.Dataset.Front (); sample != nil; sample = sample.Next () {
		sv = sample.Value.(*dsv.RecordSlice)

		if instance == sv {
			continue
		}

		// compute euclidian distance
		d = 0.0
		for i = range (*sv) {
			if i == input.ClassIdx {
				// skip class attribute
				continue
			}

			ir = &(*instance)[i]
			sr = &(*sv)[i]

			switch ir.T {
			case dsv.TReal:
				d += math.Abs (ir.Value ().(float64) - sr.Value ().(float64))
			case dsv.TInteger:
				d += math.Abs (float64(ir.Value ().(int64) - sr.Value ().(int64)))
			}
		}

		distances = append (distances, Distance { sv, math.Sqrt (d) })
	}

	sort.Sort (&distances)

	return &distances, e
}

/*
Neighbors return the nearest neighbors as pointer to slice of distance.
*/
func (input *Input) Neighbors (instance *dsv.RecordSlice) (*DistanceSlice, error) {
	var e error
	var d *DistanceSlice
	var kneighbors DistanceSlice

	switch (input.Method) {
	case TEuclidianDistance:
		d, e = input.ComputeEuclidianDistance (instance)
	}

	if nil != e {
		return nil, e
	}

	kneighbors = (*d)[0:input.K]

	return &kneighbors, e
}
