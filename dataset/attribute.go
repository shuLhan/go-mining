// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dataset

import (
	"errors"
	"fmt"
	"math"
)

/*
Attr define a simple form of all dataset in continuous or discrete value.
*/
type Attr struct {
	// IsContinu indicated whether the attribute is continuous or not.
	IsContinu bool
	// Values contain the interface to slice of values.
	Values interface{}
	// NominalValue contain possible value in discrete attribute.
	NominalValues []string
}

/*
GetContinuValues return the attribute value for continuous type.
*/
func (attr *Attr) GetContinuValues() *[]float64 {
	if attr.IsContinu {
		return attr.Values.(*[]float64)
	}
	return nil
}

/*
GetDiscreteValues return the attribute value for discrete type.
*/
func (attr *Attr) GetDiscreteValues() *[]string {
	if attr.IsContinu {
		return nil
	}
	return attr.Values.(*[]string)
}

/*
GetNominalValue return the nominal value for discrete attribute.
If attribute is continuous, return nil.
*/
func (attr *Attr) GetNominalValue() []string {
	if attr.IsContinu {
		return nil
	}
	return attr.NominalValues
}

/*
Split the attribute value, move the attribute value from start to end to the
new split and keep the remaining value in current attr.
*/
func (attr *Attr) Split(start, end int) (split Attr, e error) {
	if start < 0 {
		return split, errors.New("Split: Invalid start value.")
	}

	var size int
	var attrC *[]float64
	var attrD *[]string

	if attr.IsContinu {
		attrC = attr.GetContinuValues()
		size = len(*attrC)
	} else {
		attrD = attr.GetDiscreteValues()
		size = len(*attrD)
	}

	if end > size {
		end = size
	}

	split.IsContinu = attr.IsContinu
	split.NominalValues = attr.NominalValues

	if attr.IsContinu {
		c := make([]float64, end - start)
		copy(c, (*attrC)[start:end])
		split.Values = &c

		(*attrC) = append((*attrC)[:start], (*attrC)[end:]...)
	} else {
		c := make([]string, end - start)
		copy(c, (*attrD)[start:end])
		split.Values = &c

		(*attrD) = append((*attrD)[:start], (*attrD)[end:]...)
	}

	return
}

/*
CountNominal will count each nominal values in attribute values.

DO NOT USE MAP! The result is unpredictable.
*/
func (attr *Attr) CountNominal() (*[]int) {
	nominalCnt := make([]int, len(attr.NominalValues))
	values := attr.GetDiscreteValues()

	for _,a := range (*values) {
		for k,v := range attr.NominalValues {
			if a == v {
				nominalCnt[k]++
				break
			}
		}
	}

	return &nominalCnt
}

/*
NominalGetMax find the nominal value with highest count.
*/
func (attr *Attr) NominalGetMax(nominalCnt *[]int) (maxK string, maxV int) {
	maxV = 0
	for k,v := range (*nominalCnt) {
		if v > maxV {
			maxK = attr.NominalValues[k]
			maxV = v
		}
	}

	return
}

/*
NominalGetMin find the nominal value with lowest count.
*/
func (attr *Attr) NominalGetMin(nominalCnt *[]int) (minK string, minV int) {
	minV = math.MaxInt32

	for k,v := range (*nominalCnt) {
		if v < minV {
			minK = attr.NominalValues[k]
			minV = v
		}
	}

	return
}

/*
GetNominalMajority return majority nominal value and their number in attribute
values.
*/
func (attr *Attr) GetNominalMajority() (string,int) {
	nominalCnt := attr.CountNominal()

	return attr.NominalGetMax(nominalCnt)
}

/*
GetNominalMinority return nominal value and their number in attribute
values with the lowest count.
*/
func (attr *Attr) GetNominalMinority() (string,int) {
	nominalCnt := attr.CountNominal()

	return attr.NominalGetMin(nominalCnt)
}

/*
ClearValues set the attribute value to empty.
*/
func (attr *Attr) ClearValues() {
	if attr.IsContinu {
		attrC := attr.GetContinuValues()

		for i := range *attrC {
			(*attrC)[i] = 0.0
		}
	} else {
		attrD := attr.GetDiscreteValues()

		for i := range *attrD {
			(*attrD)[i] = ""
		}
	}
}

/*
String return the pretty print format of attribute.
*/
func (attr Attr) String() (s string) {
	s = fmt.Sprintf("{\n\tIsContinue: %v\n", attr.IsContinu)

	s += fmt.Sprintf("\tValues: %v\n", attr.Values)

	if ! attr.IsContinu {
		s += fmt.Sprintf("\tNominalValues: %v\n", attr.NominalValues)
	}

	s += "}"

	return
}
