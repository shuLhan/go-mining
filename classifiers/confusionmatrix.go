// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package classifiers

import (
	"fmt"
	"github.com/shuLhan/tabula"
	"os"
	"strconv"
)

var (
	// DEBUG level.
	DEBUG = 0
)

/*
ConfusionMatrix represent the matrix of classification.
*/
type ConfusionMatrix struct {
	tabula.Dataset
	// rowNames contain name in each row.
	rowNames []string
	// nSamples contain number of class.
	nSamples int64
	// nTrue contain number of true positive and negative.
	nTrue int64
	// nFalse contain number of false positive and negative.
	nFalse int64
}

func init() {
	v := os.Getenv("CONFUSIONMATRIX_DEBUG")
	if v == "" {
		DEBUG = 0
	} else {
		DEBUG, _ = strconv.Atoi(v)
	}
}

/*
NewConfusionMatrix create and return new confusion matrix using `valueSpace`,
`targets`, and `predictions`.
*/
func NewConfusionMatrix(valueSpace, targets, predictions []string) (
	cm *ConfusionMatrix,
) {
	cm = &ConfusionMatrix{}

	cm.compute(valueSpace, targets, predictions)

	return cm
}

/*
compute will calculate confusion matrix using targets and predictions
class values.
*/
func (cm *ConfusionMatrix) compute(valueSpace, targets, predictions []string) {
	cm.init(valueSpace)

	for x, target := range valueSpace {
		col := cm.GetColumn(x)

		for _, predict := range valueSpace {
			cnt := cm.countTargetPrediction(target, predict,
				targets, predictions)

			rec := tabula.Record{V: cnt}
			col.PushBack(&rec)
		}

		cm.PushColumnToRows(*col)
	}

	cm.computeClassError()

	if DEBUG >= 2 {
		fmt.Println("[randomforest]", &cm)
	}

	return
}

/*
create will initialize confusion matrix using value space.
*/
func (cm *ConfusionMatrix) init(valueSpace []string) {
	var colTypes []int
	var colNames []string

	for _, v := range valueSpace {
		colTypes = append(colTypes, tabula.TInteger)
		colNames = append(colNames, v)
		cm.rowNames = append(cm.rowNames, v)
	}

	// class error column
	colTypes = append(colTypes, tabula.TReal)
	colNames = append(colNames, "class_error")

	cm.Dataset.Init(tabula.DatasetModeMatrix, colTypes, colNames)
}

/*
countTargetPrediction will count and return number of true positive or false
positive in predictions using targets values.
*/
func (cm *ConfusionMatrix) countTargetPrediction(target, predict string,
	targets, predictions []string,
) (
	cnt int64,
) {
	predictslen := len(predictions)

	for x, v := range targets {
		// In case out of range, where predictions length less than
		// targets length.
		if x > predictslen {
			break
		}
		if v != target {
			continue
		}
		if predict == predictions[x] {
			cnt++
		}
	}
	return
}

/*
computeClassError will compute the classification error in matrix.
*/
func (cm *ConfusionMatrix) computeClassError() {
	var tp, fp int64

	cm.nSamples = 0
	cm.nFalse = 0

	classcol := cm.GetNColumn() - 1
	col := cm.GetColumnClassError()
	rows := cm.GetDataAsRows()
	for x, row := range *rows {
		for y, cell := range row {
			if y == classcol {
				break
			}
			if x == y {
				tp = cell.Integer()
			} else {
				fp += cell.Integer()
			}
		}

		nSamplePerRow := tp + fp
		errv := float64(fp) / float64(nSamplePerRow)
		rec := tabula.Record{V: errv}
		col.PushBack(&rec)

		cm.nSamples += nSamplePerRow
		cm.nTrue += tp
		cm.nFalse += fp
	}

	cm.PushColumnToRows(*col)
}

/*
GetColumnClassError return the last column which is the column that contain
the error of classification.
*/
func (cm *ConfusionMatrix) GetColumnClassError() *tabula.Column {
	return cm.GetColumn(cm.GetNColumn() - 1)
}

/*
GetTPRate return true-positive rate in term of

	true-positive / (true-positive + false-positive)
*/
func (cm *ConfusionMatrix) GetTrueRate() float64 {
	return float64(cm.nTrue) / float64(cm.nTrue+cm.nFalse)
}

/*
GetFPRate return false-positive rate in term of,

	false-positive / (false-positive + true negative)
*/
func (cm *ConfusionMatrix) GetFalseRate() float64 {
	return float64(cm.nFalse) / float64(cm.nTrue+cm.nFalse)
}

/*
String will return the output of confusion matrix in table like format.
*/
func (cm *ConfusionMatrix) String() (s string) {
	s += "Confusion Matrix:\n"

	// Row header: column names.
	s += "\t"
	for _, col := range cm.GetColumnsName() {
		s += col + "\t"
	}
	s += "\n"

	rows := cm.GetDataAsRows()
	for x, row := range *rows {
		s += cm.rowNames[x] + "\t"

		for _, v := range row {
			s += v.String() + "\t"
		}
		s += "\n"
	}

	return
}
