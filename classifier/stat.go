// Copyright 2015-2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package classifier

import (
	"github.com/shuLhan/tabula"
)

/*
Stat hold statistic value of classifier, including TP rate, FP rate, precision,
and recall.
*/
type Stat struct {
	// ID unique id for this statistic (e.g. number of tree).
	ID int64
	// StartTime contain the start time of classifier in unix timestamp.
	StartTime int64
	// EndTime contain the end time of classifier in unix timestamp.
	EndTime int64
	// ElapsedTime contain actual time, in seconds, between end and start
	// time.
	ElapsedTime int64
	// TP contain true-positive value.
	TP int64
	// FP contain false-positive value.
	FP int64
	// TN contain true-negative value.
	TN int64
	// FN contain false-negative value.
	FN int64
	// OobError contain out-of-bag error.
	OobError float64
	// OobErrorMean contain mean of out-of-bag error.
	OobErrorMean float64
	// TPRate contain true-positive rate (recall): tp/(tp+fn)
	TPRate float64
	// FPRate contain false-positive rate: fp/(fp+tn)
	FPRate float64
	// TNRate contain true-negative rate: tn/(tn+fp)
	TNRate float64
	// Precision contain: tp/(tp+fp)
	Precision float64
	// FMeasure contain value of F-measure or the harmonic mean of
	// precision and recall.
	FMeasure float64
	// Accuracy contain the degree of closeness of measurements of a
	// quantity to that quantity's true value.
	Accuracy float64
}

/*
Recall return value of recall.
*/
func (stat *Stat) Recall() float64 {
	return stat.TPRate
}

//
// Sum will add statistic from other stat object to current stat, not including
// the start and end time.
//
func (stat *Stat) Sum(other *Stat) {
	stat.OobError += other.OobError
	stat.OobErrorMean += other.OobErrorMean
	stat.TP += other.TP
	stat.FP += other.FP
	stat.TN += other.TN
	stat.FN += other.FN
	stat.TPRate += other.TPRate
	stat.FPRate += other.FPRate
	stat.TNRate += other.TNRate
	stat.Precision += other.Precision
	stat.FMeasure += other.FMeasure
	stat.Accuracy += other.Accuracy
}

//
// ToRow will convert the stat to tabula.row in the order of Stat field.
//
func (stat *Stat) ToRow() (row *tabula.Row) {
	row = &tabula.Row{}

	row.PushBack(tabula.NewRecordInt(stat.ID))
	row.PushBack(tabula.NewRecordInt(stat.StartTime))
	row.PushBack(tabula.NewRecordInt(stat.EndTime))
	row.PushBack(tabula.NewRecordInt(stat.ElapsedTime))
	row.PushBack(tabula.NewRecordReal(stat.OobError))
	row.PushBack(tabula.NewRecordReal(stat.OobErrorMean))
	row.PushBack(tabula.NewRecordInt(stat.TP))
	row.PushBack(tabula.NewRecordInt(stat.FP))
	row.PushBack(tabula.NewRecordInt(stat.TN))
	row.PushBack(tabula.NewRecordInt(stat.FN))
	row.PushBack(tabula.NewRecordReal(stat.TPRate))
	row.PushBack(tabula.NewRecordReal(stat.FPRate))
	row.PushBack(tabula.NewRecordReal(stat.TNRate))
	row.PushBack(tabula.NewRecordReal(stat.Precision))
	row.PushBack(tabula.NewRecordReal(stat.FMeasure))
	row.PushBack(tabula.NewRecordReal(stat.Accuracy))

	return
}
