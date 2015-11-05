package dataset

import (
	"errors"
	"github.com/shuLhan/dsv"
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
IsInSingleClass check wether all target class contain only single class.
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
