// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package dataset extend the dsv.Reader to read data from delimited separated
value (DSV) file by adding attribute ClassIndex which indicate the index of
target attribute (class) in data.

This package also extends the metadata to include additional attribute
IsContinue (to indicate whether the field is continuous) and NominalValues
(which contain the discrete values).
*/
package dataset

import (
	"encoding/json"
	"log"

	"github.com/shuLhan/dsv"
)

/*
Reader contain metadata for reading input file, metadata for each column,
and the data.
*/
type Reader struct {
	// Reader extend dsv.Reader
	dsv.Reader
	// InputMetadata with additional field.
	InputMetadata	[]Metadata	`json:"InputMetadata"`
	// ClassIndex for target classification in fields attributes.
	ClassIndex	int		`json:"ClassIndex"`
}

/*
NewReader create new dataset from 'input' configuration.
*/
func NewReader (input string) (ds *Reader, e error) {
	ds = &Reader {}

	e = dsv.Open (ds, input)

	if nil != e {
		return nil, e
	}

	return ds, e
}

/*
Init initialize reader for dataset.
*/
func (dataset *Reader) Init () (e error) {
	dataset.Reader.OutputMode = dataset.OutputMode

	// Create reader input metadata for reading.
	dataset.Reader.InputMetadata = make ([]dsv.Metadata, len (dataset.InputMetadata))
	for i := range dataset.InputMetadata {
		dataset.Reader.InputMetadata[i] = dataset.InputMetadata[i].Metadata
	}

	return dataset.Reader.Init ()
}

/*
GetInputMetadata return metadata which required by ReaderInterface.
*/
func (dataset *Reader) GetInputMetadata () *[]dsv.Metadata {
	return &dataset.Reader.InputMetadata
}

/*
GetInputMetadataAt return single metadata which required by function Open in
ReaderInterface.
*/
func (dataset *Reader) GetInputMetadataAt (idx int) *dsv.Metadata {
	return &dataset.Reader.InputMetadata[idx]
}

/*
GetClassCount return number of classification.
*/
func (dataset *Reader) GetClassCount () int {
	return len (dataset.InputMetadata[dataset.ClassIndex].NominalValues)
}

/*
GetClassValues return the classification values.
*/
func (dataset *Reader) GetClassValues () []string {
	return dataset.InputMetadata[dataset.ClassIndex].NominalValues;
}

/*
String yes, it will print it in JSON like format.
*/
func (dataset *Reader) String() string {
	r, e := json.MarshalIndent (dataset, "", "\t")

	if nil != e {
		log.Println (e)
	}

	return string (r)
}
