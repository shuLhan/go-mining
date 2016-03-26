// Copyright 2015-2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package classifiers

import (
	"time"
)

/*
Stat hold statistic value of classifier, including TP rate, FP rate, precision,
and recall.
*/
type Stat struct {
	// startTime contain the start time of classifier.
	StartTime time.Time
	// endTime contain the end time of classifier.
	EndTime time.Time
	// TPRate contain true-positive rate (recall): tp/(tp+fn)
	TPRate float64
	// FPRate contain false-positive rate: fp/(fp+tn)
	FPRate float64
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
