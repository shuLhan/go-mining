// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package dataset extend the dsv.Reader to read data from delimited separated
value (DSV) file by adding attribute ClassIndex which indicate the index of
target attribute (class) in data.

This package also extends the metadata to include additional attribute
IsContinue (to indicate whether the column is continuous) and NominalValues
(which contain the discrete values).
*/
package dataset

import (
	"encoding/json"
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/dsv/util"
	"io"
	"log"
)

/*
Reader contain metadata for reading input file, metadata for each column,
and the data.
*/
type Reader struct {
	// Reader embed dsv.Reader
	dsv.Reader
	// InputMetadata with additional attributes.
	InputMetadata []Metadata `json:"InputMetadata"`
	// ClassIndex for target classification in columns attributes.
	ClassIndex int `json:"ClassIndex"`
	// MajorityClass contain the name of class that have majority data.
	MajorityClass string
	// MinorityClass contain the name of class that have minority data.
	MinorityClass string
}

/*
NewReader create new dataset from 'input' configuration.
*/
func NewReader(config string) (reader *Reader, e error) {
	reader = &Reader{}

	e = dsv.OpenReader(reader, config)

	if nil != e {
		return nil, e
	}

	return reader, e
}

/*
GetInputMetadata return input metadata. Since we override the input metadata,
we must return our custom input metadata back using this function.
*/
func (reader *Reader) GetInputMetadata() []dsv.MetadataInterface {
	md := make([]dsv.MetadataInterface, len(reader.InputMetadata))
	for i := range reader.InputMetadata {
		md[i] = &reader.InputMetadata[i]
	}

	return md
}

/*
Read a wrapper for dsv reader with additional post-read initialization like
counting majority and minority class.
*/
func (reader *Reader) Read() (e error) {
	_, e = dsv.Read(reader)

	if e != nil && e != io.EOF {
		return
	}

	targetV := reader.GetTarget().ToStringSlice()
	classV := reader.GetClass()

	classCount := util.StringCountBy(targetV, classV)

	_, maxIdx := util.IntFindMax(classCount)
	_, minIdx := util.IntFindMin(classCount)
	reader.MajorityClass = classV[maxIdx]
	reader.MinorityClass = classV[minIdx]

	return
}

/*
GetClass return the classification values.
*/
func (reader *Reader) GetClass() []string {
	return reader.InputMetadata[reader.ClassIndex].NominalValues
}

/*
GetTargetMetadata return the target attribute of input.
*/
func (reader *Reader) GetTargetMetadata() Metadata {
	return reader.InputMetadata[reader.ClassIndex]
}

/*
GetTarget return the target values in column.
*/
func (reader *Reader) GetTarget() *dsv.Column {
	return &reader.Columns[reader.ClassIndex]
}

/*
GetMajorityClass return the majority class of data.
*/
func (reader *Reader) GetMajorityClass() string {
	return reader.MajorityClass
}

/*
IsInSingleClass check whether all target class contain only single class.
Return true and name of target if all rows is in the same class,
false and empty string otherwise.
*/
func (reader *Reader) IsInSingleClass() (single bool, class string) {
	target := reader.GetTarget().ToStringSlice()

	for i, t := range target {
		if i == 0 {
			single = true
			class = t
			continue
		}
		if t != class {
			return false, ""
		}
	}
	return
}

/*
SplitRowsByValue will split dataset by column value.
This function overwrite the dsv.dataset, by returning the Reader object
instead of dataset object.
*/
func (reader *Reader) SplitRowsByValue(colidx int, colval interface{}) (
	splitL *Reader,
	splitR *Reader,
	e error,
) {
	splitL = &Reader{}
	splitR = &Reader{}

	splitL.Dataset, splitR.Dataset, e = reader.Dataset.SplitRowsByValue(
		colidx, colval)

	if e != nil {
		return nil, nil, e
	}

	splitL.InputMetadata = make([]Metadata, len(reader.InputMetadata))
	copy(splitL.InputMetadata, reader.InputMetadata)

	splitR.InputMetadata = make([]Metadata, len(reader.InputMetadata))
	copy(splitR.InputMetadata, reader.InputMetadata)

	splitL.ClassIndex = reader.ClassIndex
	splitR.ClassIndex = reader.ClassIndex

	splitL.TransposeToColumns()
	splitR.TransposeToColumns()

	return
}

/*
String yes, it will print it in JSON like format.
*/
func (reader *Reader) String() string {
	r, e := json.MarshalIndent(reader, "", "\t")

	if nil != e {
		log.Println(e)
	}

	return string(r)
}

/*
PrintTable will return data formatted in table.
*/
func (reader *Reader) PrintTable() (s string) {
	for a := range reader.InputMetadata {
		s += fmt.Sprintf("\t[%d]", a)
	}
	s += "\nCont:"
	for a := range reader.InputMetadata {
		s += fmt.Sprint("\t", reader.InputMetadata[a].IsContinu)
	}
	s += "\n"

	for i := 0; i < reader.NRow; i++ {
		s += fmt.Sprintf("[%d]", i)

		for a, md := range reader.InputMetadata {
			if md.IsContinu {
				s += fmt.Sprint("\t",
					reader.Columns[a][i].Float())
			} else {
				s += fmt.Sprint("\t",
					reader.Columns[a][i].String())
			}
		}
		s += "\n"
	}

	return
}
