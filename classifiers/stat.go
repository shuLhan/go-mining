// Copyright 2015-2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package classifiers

/*
Stat hold statistic value of classifier, including TP rate, FP rate, precision,
and recall.
*/
type Stat struct {
	// TPRate contain true-positive rate: tp/(tp+fn)
	TPRate float64
	// FPRate contain false-positive rate: fp/(fp+tn)
	FPRate float64
	// Precision contain: tp/(tp+fp)
	Precision float64
}

/*
Recall return value of recall.
*/
func (stat *Stat) Recall() float64 {
	return stat.TPRate
}
