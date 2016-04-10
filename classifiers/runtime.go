// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package classifiers

import (
	"github.com/shuLhan/dsv"
)

//
// Runtime define a generic type which provide common fields that can be
// embedded by the real classifier (e.g. RandomForest).
//
type Runtime struct {
	// StatsFile is the file where performance statistic will be written.
	StatsFile string `json:"StatsFile"`

	// cmatrices contain confusion matrix value for each iteration.
	cmatrices []ConfusionMatrix

	// stats contain statistic of classifier for each stage.
	stats Stats

	// StatTotal contain total statistic values.
	statTotal Stat

	// statWriter contain file writer for statistic.
	statWriter *dsv.Writer
}

//
// Stats return all statistic objects.
//
func (runtime *Runtime) Stats() *Stats {
	return &runtime.stats
}

//
// StatTotal return total statistic.
//
func (runtime *Runtime) StatTotal() *Stat {
	return &runtime.statTotal
}

//
// AddConfusionMatrix will append new confusion matrix.
//
func (runtime *Runtime) AddConfusionMatrix(cm *ConfusionMatrix) {
	runtime.cmatrices = append(runtime.cmatrices, *cm)
}

//
// AddStat will append new classifier statistic data.
//
func (runtime *Runtime) AddStat(stat *Stat) {
	runtime.stats = append(runtime.stats, stat)
}

//
// ComputeStatTotal compute total statistic.
//
func (runtime *Runtime) ComputeStatTotal(stat *Stat) {
	if stat == nil {
		return
	}

	nstat := len(runtime.stats)
	if nstat == 0 {
		return
	}

	t := &runtime.statTotal

	t.OobError += stat.OobError
	t.OobErrorMean = t.OobError / float64(nstat)
	t.TP += stat.TP
	t.FP += stat.FP
	t.TN += stat.TN
	t.FN += stat.FN
	t.TPRate = float64(t.TP) / float64(t.TP+t.FN)
	t.FPRate = float64(t.FP) / float64(t.FP+t.TN)
	t.TNRate = float64(t.TN) / float64(t.FP+t.TN)
	t.Precision = float64(t.TP) / float64(t.TP+t.FP)
	t.FMeasure = 2 / ((1 / t.Precision) + (1 / t.TPRate))
	t.Accuracy = float64(t.TP+t.TN) / float64(t.TP+t.TN+t.FP+t.FN)
}

//
// OpenStatsFile will open statistic file for output.
//
func (runtime *Runtime) OpenStatsFile() error {
	if runtime.statWriter != nil {
		_ = runtime.CloseStatsFile()
	}
	runtime.statWriter = &dsv.Writer{}
	return runtime.statWriter.OpenOutput(runtime.StatsFile)
}

//
// WriteStat will write statistic of process to file.
//
func (runtime *Runtime) WriteStat(stat *Stat) error {
	if runtime.statWriter == nil {
		return nil
	}
	if stat == nil {
		return nil
	}
	return runtime.statWriter.WriteRawRow(stat.ToRow(), nil, nil)
}

//
// CloseStatsFile will close statistics file for writing.
//
func (runtime *Runtime) CloseStatsFile() (e error) {
	if runtime.statWriter == nil {
		return
	}

	e = runtime.statWriter.Close()
	runtime.statWriter = nil

	return
}
