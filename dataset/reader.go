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
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/dsv/util"
	"io"
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
	// ClassMetadataIndex target classification in metadata
	ClassMetadataIndex int `json:"ClassMetadataIndex"`
	// ClassIndex for target classification in columns attributes.
	// It could be different with ClassMetadataIndex because of Skip value
	// in certain column.
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
ReaderCopy set reader attribute value using other reader value.
*/
func (reader *Reader) ReaderCopy(src *Reader) {
	if src == nil {
		return
	}

	mdlen := len(src.InputMetadata)
	reader.InputMetadata = make([]Metadata, mdlen)

	for x := 0; x < mdlen; x++ {
		reader.InputMetadata[x] = src.InputMetadata[x].CreateCopy()
	}

	reader.ClassIndex = src.ClassIndex
	reader.ClassMetadataIndex = src.ClassMetadataIndex
	reader.MajorityClass = src.MajorityClass
	reader.MinorityClass = src.MinorityClass
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
CountMajorMinorClass recount major and minor class in dataset.
*/
func (reader *Reader) CountMajorMinorClass() {
	targetV := reader.GetTarget().ToStringSlice()
	classV := reader.GetClass()

	classCount := util.StringCountBy(targetV, classV)

	_, maxIdx := util.IntFindMax(classCount)
	_, minIdx := util.IntFindMin(classCount)
	reader.MajorityClass = classV[maxIdx]
	reader.MinorityClass = classV[minIdx]
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

	reader.CountMajorMinorClass()

	return
}

/*
GetClass return the classification values.
*/
func (reader *Reader) GetClass() []string {
	return reader.InputMetadata[reader.ClassMetadataIndex].NominalValues
}

/*
GetTargetMetadata return the target attribute of input.
*/
func (reader *Reader) GetTargetMetadata() Metadata {
	return reader.InputMetadata[reader.ClassMetadataIndex]
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

	splitL.ReaderCopy(reader)
	splitR.ReaderCopy(reader)

	splitL.CountMajorMinorClass()
	splitR.CountMajorMinorClass()

	return
}

/*
PushColumn append new column to dataset including their metadata.
*/
func (reader *Reader) PushColumn(col dsv.Column, md Metadata) {
	reader.Dataset.PushColumn(col)
	reader.InputMetadata = append(reader.InputMetadata, md)
}

/*
RandomPickRows return `n` rows that randomly picked from dataset.
*/
func (reader *Reader) RandomPickRows(n int, dup bool) (
	picked Reader,
	unpicked Reader,
	pickedIdx []int,
	unpickedIdx []int,
	e error,
) {
	picked.Dataset, unpicked.Dataset, pickedIdx, unpickedIdx, e =
		reader.Reader.RandomPickRows(n, dup)

	picked.ReaderCopy(reader)
	picked.CountMajorMinorClass()

	unpicked.ReaderCopy(reader)
	unpicked.CountMajorMinorClass()

	return
}

/*
RandomPickColumns return `n` column that randomly picked from dataset.
If dup is true, then column that has been picked can be pick up gain.
otherwise one column can only be pick up once.
*/
func (reader *Reader) RandomPickColumns(n int, dup bool) (
	picked Reader,
	unpicked Reader,
	pickedIdx []int,
	unpickedIdx []int,
	e error,
) {
	// exclude target column
	excludeIdx := []int{reader.ClassIndex}

	picked.Dataset, unpicked.Dataset, pickedIdx, unpickedIdx, e =
		reader.Reader.RandomPickColumns(n, dup, excludeIdx)

	if e != nil {
		return
	}

	// set picked metadata
	var mds []Metadata

	for _, v := range pickedIdx {
		mds = append(mds, reader.InputMetadata[v])
	}
	picked.InputMetadata = mds

	// set unpicked metadata
	mds = make([]Metadata, 0)
	for _, v := range unpickedIdx {
		mds = append(mds, reader.InputMetadata[v])
	}
	unpicked.InputMetadata = mds

	targetAttr := reader.GetTarget()
	targetMd := reader.GetTargetMetadata()

	// add target column to picked set
	picked.PushColumn(*targetAttr, targetMd)

	classIdx := len(picked.Columns) - 1
	picked.ClassMetadataIndex = classIdx
	picked.ClassIndex = classIdx

	// add target column to unpicked set
	unpicked.PushColumn(*targetAttr, targetMd)

	classIdx = len(unpicked.Columns) - 1
	unpicked.ClassMetadataIndex = classIdx
	unpicked.ClassIndex = classIdx

	picked.CountMajorMinorClass()
	unpicked.CountMajorMinorClass()

	return
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

	for i := 0; i < reader.GetNRow(); i++ {
		s += fmt.Sprintf("[%d]", i)

		for a, md := range reader.InputMetadata {
			if md.IsContinu {
				s += fmt.Sprint("\t",
					reader.Columns[a].Records[i].Float())
			} else {
				s += fmt.Sprint("\t",
					reader.Columns[a].Records[i].String())
			}
		}
		s += "\n"
	}

	return
}

func (reader *Reader) String() (s string) {
	s = fmt.Sprint("{\n",
	"  InputMetadata     : ", reader.InputMetadata, "\n",
	"  ClassMetadataIndex: ", reader.ClassMetadataIndex, "\n",
	"  ClassIndex        : ", reader.ClassIndex, "\n",
	"  NRow              : ", reader.GetNRow(), "\n",
	"  Dataset           : ", fmt.Sprint(reader.Dataset), "\n",
	"  MajorityClass     : ", reader.MajorityClass, "\n",
	"  MinorityClass     : ", reader.MinorityClass, "\n",
	"}")

	return
}
