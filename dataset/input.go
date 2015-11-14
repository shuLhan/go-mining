// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dataset

import (
	"errors"
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/dsv/util"
	"github.com/shuLhan/go-mining/set"
)

/*
Input a simplified, converted, value of dataset.
*/
type Input struct {
	// Attrs is a slice of all attribute in dataset.
	Attrs []Attr
	// Size of slice in attribute value, should be the same in for all
	// attributes.
	Size int
	// ClassIdx is an index to target class attribute.
	ClassIdx int
	// ClassCnt contain the number of class in target attribute.
	ClassCnt *[]int
	// MajorityClass contain the name of class that have majority data.
	MajorityClass string
	// MinorityClass contain the name of class that have minority data.
	MinorityClass string
}

/*
NewInput will convert all fields in dataset to their type (continuous or
discrete) and merge the nominal value from metadata.
*/
func NewInput(dataset *[]dsv.RecordSlice, md *[]Metadata, classIdx int) (
							in *Input, e error) {
	mdSize := len(*md)

	if len(*dataset) < mdSize {
		return nil,
		errors.New("Number of metadata is greater than dataset.")
	}
	if classIdx < 0 || classIdx > mdSize {
		return nil,
		errors.New("Class index is out of range.")
	}

	in = &Input{
		Attrs: make([]Attr, len(*md)),
		Size: len((*dataset)[0]),
		ClassIdx: classIdx,
	}

	for i,m := range (*md) {
		attr := Attr{}

		attr.IsContinu = m.IsContinu
		attr.NominalValues = m.NominalValues

		if m.IsContinu {
			attr.Values = dsv.RecordSliceToFloatSlice(
							&(*dataset)[i])
		} else {
			attr.Values = dsv.RecordSliceToStringSlice(
							&(*dataset)[i])
		}

		in.Attrs[i] = attr
	}

	// count number of each class in target attribute.
	targetAttr := in.GetTargetAttr()
	in.ClassCnt = targetAttr.CountNominal()

	// calculate the majority and minority class.
	in.MajorityClass,_ = targetAttr.NominalGetMax(in.ClassCnt)
	in.MinorityClass,_ = targetAttr.NominalGetMin(in.ClassCnt)

	return
}

/*
GetMajorityClass return the majority class of data.
*/
func (in *Input) GetMajorityClass() string {
	return in.MajorityClass
}

/*
GetTargetAttr return the target attribute of input.
*/
func (in *Input) GetTargetAttr() (*Attr) {
	return &in.Attrs[in.ClassIdx]
}

/*
GetTargetAttrValues return the slice of target attribute.
*/
func (in *Input) GetTargetAttrValues() (*[]string) {
	return in.Attrs[in.ClassIdx].GetDiscreteValues()
}

/*
IsInSingleClass check whether all target class contain only single class.
*/
func (in *Input) IsInSingleClass() (single bool) {
	var class string
	target := in.GetTargetAttrValues()

	for i,t := range (*target) {
		if i == 0 {
			single = true
			class = t
			continue
		}
		if t != class {
			return false
		}
	}

	return
}

/*
Split return chunk of dataset from index start to index end.
The original data in input will be changed, excluding all data that has been
splited.
*/
func (in *Input) Split(start, end int) (*Input, error) {
	if start < 0 {
		return nil, errors.New("Split: Invalid start value.")
	}

	if end > in.Size {
		end = in.Size
	}

	var e error
	var split *Input

	split = new(Input)

	split.Attrs = make([]Attr, len(in.Attrs))
	split.Size = end - start
	split.ClassIdx = in.ClassIdx
	split.MajorityClass = in.MajorityClass
	split.MinorityClass = in.MinorityClass

	for i,a := range in.Attrs {
		split.Attrs[i],e = a.Split(start, end)

		if e != nil {
			return nil,e
		}
	}

	in.Size = in.Size - (end - start)

	return split,nil
}

/*
SplitByAttrValue will split all attribute using attrIdx as split reference,
and using attrV as the value for splitting.
*/
func (in *Input) SplitByAttrValue(attrIdx int, attrV interface{}) (
							split *Input) {
	if in.Attrs[attrIdx].IsContinu {
		attrC := attrV.(float64)
		split = in.SplitAttrContinu(attrIdx, attrC)
	} else {
		attrD := attrV.(set.SliceString)
		split = in.SplitAttrDiscrete(attrIdx, attrD)
	}

	return split
}

/*
SplitAttrContinu will split all attribute by value of attrIdx.

NOTE: This function assume that all attributes has been sorted according to
attrIdx.

If the attribute referenced by attrIdx is continuous, all the attribute value
less than attrV will be moved to new dataset, and the remains will keep in
current dataset. For example, given two continuous attribute,

	A: {1,2,3,4}
	B: {5,6,7,8}

if attrIdx is B and attrV is 6.5, the splitted input will be,

	A': {1,2}
	B': {5,6}

and the remaining input would be,

	A: {3,4}
	B: {7,8}
*/
func (in *Input) SplitAttrContinu(attrIdx int, attrV float64) (*Input) {
	splitAttr := in.Attrs[attrIdx].GetContinuValues()
	splitIdx := len(*splitAttr)

	// find the index where attribute value is greater than attrV.
	for i := range *splitAttr {
		if (*splitAttr)[i] > attrV {
			splitIdx = i
			break
		}
	}

	// and then split all attributes using split index.
	split, e := in.Split(0, splitIdx)

	if e != nil {
		// there are no possibilities that the split range is wrong.
		fmt.Println("SplitAttrContinu: error ", e)
		return nil
	}

	return split
}

/*
SplitAttrDiscrete will split all attribute based on nominal value in attrV.

If the attribute referenced by attrIdx is discrete, select the row that
has the attrV in referenced attribute and move it to the new dataset, and keep
the rest in current dataset. For example, given two attribute,

	A: {1,2,3,4}
	B: {A,B,A,C}

if attrIdx is B and attrV is A, the splitted input will be,

	A':{1,3}
	B':{A,A}

while the remaining input would be,

	A: {2,4}
	B: {B,C}
*/
func (in *Input) SplitAttrDiscrete(attrIdx int, attrD set.SliceString) (
							split *Input) {
	split = new(Input)

	split.Attrs = make([]Attr, len(in.Attrs))
	split.ClassIdx = in.ClassIdx
	split.MajorityClass = in.MajorityClass
	split.MinorityClass = in.MinorityClass

	// Get the index of attrD in attrIdx values.
	var splitIdx []int
	splitAttr := in.Attrs[attrIdx].GetDiscreteValues()

	for i,x := range *splitAttr {
		for _,y := range attrD {
			if x == y {
				splitIdx = append(splitIdx, i)
			}
		}
	}

	// After we got the split index, we create two slice: one for new split
	// and another for input (leftover, non-indexed).
	for i := range (*in).Attrs {
		var attrC *[]float64
		var attrD *[]string
		var size int
		var attrCSplit []float64
		var attrCLeft []float64
		var attrDSplit []string
		var attrDLeft []string

		isCont := (*in).Attrs[i].IsContinu

		if isCont {
			attrC = (*in).Attrs[i].Values.(*[]float64)
			size = len(*attrC)
		} else {
			attrD = (*in).Attrs[i].Values.(*[]string)
			size = len(*attrD)
		}

		k := 0
		for _,j := range splitIdx {
			// insert non-indexed value to left-over
			for ; k < j; k++ {
				if isCont {
					attrCLeft = append(attrCLeft, (*attrC)[k])
				} else {
					attrDLeft = append(attrDLeft, (*attrD)[k])
				}
			}
			k = j + 1

			// insert indexed value to split
			if isCont {
				attrCSplit = append(attrCSplit, (*attrC)[j])
			} else {
				attrDSplit = append(attrDSplit, (*attrD)[j])
			}
		}

		for ; k < size; k++ {
			if isCont {
				attrCLeft = append(attrCLeft, (*attrC)[k])
			} else {
				attrDLeft = append(attrDLeft, (*attrD)[k])
			}
		}

		// split done.
		if isCont {
			split.Attrs[i].Values = attrCSplit
			in.Attrs[i].Values = attrCLeft
		} else {
			split.Attrs[i].Values = attrDSplit
			in.Attrs[i].Values = attrDLeft
		}
	}

	return
}

/*
SortByIndex will sort all attribute, except sortedAttr, using sorted index.
*/
func (in *Input) SortByIndex(sortedAttr int, sortedIdx *[]int) {
	for i := range (*in).Attrs {
		if (*in).Attrs[i].IsContinu {
			attrC := (*in).Attrs[i].GetContinuValues()
			util.SortFloatSliceByIndex(attrC, sortedIdx)
		} else {
			attrD := (*in).Attrs[i].GetDiscreteValues()
			util.SortStringSliceByIndex(attrD, sortedIdx)
		}
	}
}

/*
String display the input data in table like format.
*/
func (in *Input) String() (s string) {
	for a := range in.Attrs {
		s += fmt.Sprintf("\t[%d]", a)
	}
	s += "\nCont:"
	for a := range in.Attrs {
		s += fmt.Sprint("\t", in.Attrs[a].IsContinu)
	}
	s += "\n"

	for i := 0; i < in.Size; i++ {
		s += fmt.Sprintf("[%d]", i)

		for a := range in.Attrs {
			if in.Attrs[a].IsContinu {
				s += fmt.Sprint("\t", (*in.Attrs[a].Values.(*[]float64))[i])
			} else {
				s += fmt.Sprint("\t", (*in.Attrs[a].Values.(*[]string))[i])
			}
		}
		s += "\n"
	}

	return
}
