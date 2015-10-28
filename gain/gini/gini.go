// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package gini contain function to calculating Gini gain.

Gini gain, which is an impurity-based criterion that measures the divergences
between the probability distribution of the target attributes' values.
*/
package gini

import (
	"fmt"
	"os"
	"github.com/shuLhan/dsv/util"
)

var (
	// DEBUG use this to debug the package, by printing additional
	// information when running the function.
	DEBUG = bool (os.Getenv ("DEBUG") != "")
)

/*
Gini contain slice of sorted index, slice of partition values, slice of Gini
index, Gini value for all samples.
*/
type Gini struct {
	// Index of attributed sorted by values of attribute. This will be used
	// to reference the unsorted target attribute.
	SortedIndex *[]int
	// Partition
	Part []float64
	// Index of Gini.
	Index []float64
	// value of Gini index for all value in attribute.
	Value float64
	// maxPartIndex contain the index of partition which have the maximum
	// gain.
	maxPartIndex int
	// maxGainValue contain maximum gain of index.
	maxGainValue float64
}

/*
ComputeContinu Given an attribute A and the target attribute T which contain
N classes in C, compute the information gain of A.

The result of Gini partitions value and Gini indexes is saved in Part and
Index.
*/
func (gini *Gini) ComputeContinu(A *[]float64, T *[]string, C *[]string) {
	// sort the data
	if nil == gini.SortedIndex {
		gini.SortedIndex = util.IndirectSort (A)
	}

	// sort the target attribute using sorted index.
	gini.sortTarget (T)

	// create partition
	gini.createPartition (A)

	// create holder for gini index
	gini.Index = make([]float64, len(gini.Part))

	gini.Value = gini.compute (T, C)

	gini.computeGain (A, T, C)
}

/*
SortTarget attribute using sorted index.
*/
func (gini *Gini) sortTarget (T *[]string) {
	for i := range (*gini.SortedIndex) {
		if (*gini.SortedIndex)[i] > i {
			util.SwapString (T, i, (*gini.SortedIndex)[i])
		}
	}
}

/*
createPartition for dividing class and computing Gini index.
*/
func (gini *Gini) createPartition(A *[]float64) {
	l := len(*A)
	gini.Part = make([]float64, 1)

	// first partition is the first index / 2
	gini.Part[0] = (*A)[0] / 2

	// loop from first index until last index - 1
	var sum float64
	var med float64
	var exist bool
	for i := 0; i < l - 1; i++ {
		sum = (*A)[i] + (*A)[i + 1]
		med = sum / 2

		// reject if median is contained in attribute's value.
		exist = false
		for j := i; j < l; j++ {
			if (*A)[j] == med {
				exist = true
				break
			}
		}
		if ! exist {
			gini.Part = append(gini.Part, med)
		}
	}

	// last one
	gini.Part = append(gini.Part, (*A)[l-1] + 0.5)
}

/*
compute value for attribute T.

Return Gini value in the form of,

	1 - sum (probability each classes in T)
*/
func (gini *Gini) compute(T *[]string, C *[]string) float64 {
	n := float64 (len(*T))

	if n == 0 {
		return 0
	}

	classCount := make(map[string]int, len(*C))

	for _,v := range *T {
		classCount[v]++
	}

	if DEBUG {
	//	fmt.Println ("class count:", classCount)
	//	fmt.Println ("n attr:", n)
	}

	var p float64
	var sump2 float64

	for i := range classCount {
		p = float64 (classCount[i]) / n
		sump2 += (p * p)
	}

	return 1 - sump2
}

/*
computeGain for each partition.

The Gini gain formula we used here is,

	Gain(part,S) = Gini(S) - ((count(left)/S * Gini(left))
				+ (count(right)/S * Gini(right)))

where,
	- left is sub-sample from S that is less than part value.
	- right is sub-sample from S that is greater than part value.
*/
func (gini *Gini) computeGain (A *[]float64, T *[]string, C *[]string) {
	var a, nleft, nright, partidx int
	var pleft, pright float64
	var gleft, gright float64
	var tleft, tright []string

	nsample := len(*A)

	if DEBUG {
		fmt.Println ("sorted data:", A)
		fmt.Println ("Gini.Value:", gini.Value)
	}

	for p := range gini.Part {

		// find the split of samples between partition based on
		// partition value
		partidx = nsample
		for a = range *A {
			if (*A)[a] > gini.Part[p] {
				partidx = a
				break;
			}
		}

		nleft = partidx
		nright = nsample - partidx
		pleft = float64(nleft) / float64(nsample)
		pright = float64(nright) / float64(nsample)

		if partidx > 0 {
			tleft = (*T)[0:partidx]
			tright = (*T)[partidx:]

			gleft = gini.compute(&tleft, C)
			gright = gini.compute(&tright, C)
		} else {
			tleft = nil
			tright = (*T)[0:]

			gleft = 0
			gright = gini.compute(&tright, C)
		}

		if DEBUG {
			fmt.Println(tleft)
			fmt.Println(tright)

			fmt.Print ("GiniGain(", gini.Part[p],") = ",
					gini.Value, " - (")
			fmt.Print("(", pleft, " * ", gleft, ")")
			fmt.Print(" + (", pright,  " * ", gright, ")")
			fmt.Println(")")
		}

		// count class in partition
		gini.Index[p] = gini.Value - ((pleft * gleft) + (pright * gright))

		if gini.maxGainValue < gini.Index[p] {
			gini.maxGainValue = gini.Index[p]
			gini.maxPartIndex = p
		}
	}
}

/*
GetMaxPartValue return the maximum partition that have the maximum index.
*/
func (gini *Gini) GetMaxPartValue() float64 {
	return gini.Part[gini.maxPartIndex]
}

/*
GetMaxGainValue return the value of partition which contain the maximum Gini
gain.
*/
func (gini *Gini) GetMaxGainValue() float64 {
	return gini.Index[gini.maxPartIndex]
}
