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
	"encoding/json"
	"fmt"
	"log"
	"os"
	"github.com/shuLhan/dsv/util"
	"github.com/shuLhan/go-mining/set"
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
	// MaxPartIndex contain the index of partition which have the maximum
	// gain.
	MaxPartIndex int
	// MaxGainValue contain maximum gain of index.
	MaxGainValue float64
	// IsContinue define wether the Gini index came from continuous
	// attribute or not.
	IsContinu bool
	// DiscretePart contain the possible combination of discrete values.
	DiscretePart set.SetString
	// DiscreteIndex contain the Gini index value for DiscreteSet
	DiscreteIndex []float64
}

/*
ComputeDiscrete Given an attribute A with discreate value 'discval', and the
target attribute T which contain N classes in C, compute the information gain
of A.

The result is saved as gain value in MaxGainValue for each partition.
*/
func (gini *Gini) ComputeDiscrete(A *[]string, discval *[]string, T *[]string,
				C *[]string) {
	gini.IsContinu = false

	// number of samples
	nsample := float64(len(*A))

	// create partition for possible combination of discrete values.
	gini.createDiscretePartition((*discval))

	if DEBUG {
		fmt.Printf("part : %v\n", gini.DiscretePart)
	}

	gini.DiscreteIndex = make([]float64, len(gini.DiscretePart))

	// compute gini index for all samples
	gini.Value = gini.compute(T, C)

	if DEBUG {
		fmt.Println("sample:", T)
		fmt.Printf("Gini(a=%s) = %f\n",
			(*A), gini.Value)
	}

	// compute gini index for each discrete values
	for i,subPart := range gini.DiscretePart {
		// check if sub partition has at least an element
		if len(subPart) <= 0 {
			continue
		}

		sumGI := 0.0
		for _,part := range subPart {
			ndisc := 0.0
			var subT []string

			for _,el := range part {
				for t,a := range (*A) {
					if a != el {
						continue
					}

					// count how many sample with this discrete value
					ndisc++
					// split the target by discrete value
					subT = append(subT, (*T)[t])
				}
			}

			// compute gini index for subtarget
			giniIndex := gini.compute(&subT, C)

			// compute probabilites of discrete value through all samples
			p := ndisc / nsample

			probIndex := p * giniIndex

			// sum all probabilities times gini index.
			sumGI += probIndex

			if DEBUG {
				fmt.Println("subsample:", subT)
				fmt.Printf("Gini(a=%s) = %f/%f * %f = %f\n",
						part, ndisc, nsample,
						giniIndex, probIndex)
			}
		}

		gini.DiscreteIndex[i] = gini.Value - sumGI

		if gini.MaxGainValue < gini.DiscreteIndex[i] {
			gini.MaxGainValue = gini.DiscreteIndex[i]
			gini.MaxPartIndex = i
		}
	}
}

/*
createDiscretePartition will create possible combination for discrete value
in DiscretePart and their Index in DiscreteIndex.

Number of split is defined in `nsplit`.
*/
func (gini *Gini) createDiscretePartition(discval []string) {
	// no discrete values ?
	if len(discval) <= 0{
		return
	}

	// use set partition function to group the discrete values into two
	// subset.
	gini.DiscretePart = set.PartitioningSetString(discval, 2)
}

/*
ComputeContinu Given an attribute A and the target attribute T which contain
N classes in C, compute the information gain of A.

The result of Gini partitions value and Gini indexes is saved in Part and
Index.
*/
func (gini *Gini) ComputeContinu(A *[]float64, T *[]string, C *[]string) {
	gini.IsContinu = true

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

	// last one, first partition + last attribute value
	med = gini.Part[0] + (*A)[l-1]
	gini.Part = append(gini.Part, med)
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

	var p float64
	var sump2 float64

	for i := range classCount {
		p = float64(classCount[i]) / n
		sump2 += (p * p)

		if DEBUG {
			fmt.Printf(" compute (%s): (%f/%f)^2 = %f\n", *T,
					float64(classCount[i]), n, p * p)
		}

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

		if gini.MaxGainValue < gini.Index[p] {
			gini.MaxGainValue = gini.Index[p]
			gini.MaxPartIndex = p
		}
	}
}

/*
GetMaxPartValue return the maximum partition that have the maximum index.
*/
func (gini *Gini) GetMaxPartValue() float64 {
	if gini.IsContinu {
		return gini.Part[gini.MaxPartIndex]
	}

	return -1.0
}

/*
GetMaxGainValue return the value of partition which contain the maximum Gini
gain.
*/
func (gini *Gini) GetMaxGainValue() float64 {
	if gini.IsContinu {
		return gini.Index[gini.MaxPartIndex]
	}

	return gini.MaxGainValue
}

/*
String yes, it will print it JSON like format.
*/
func (gini Gini) String() (string) {
	s, e := json.MarshalIndent(gini, "", "\t")
	if nil != e {
		log.Print(e)
		return ""
	}
	return string(s)
}
