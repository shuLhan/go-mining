// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package classifier

import (
	"fmt"
	"github.com/shuLhan/dsv"
	"time"
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

	// stats contain statistic of classifier for each iteration.
	stats Stats

	// StatTotal contain total statistic values.
	statTotal Stat

	// statWriter contain file writer for statistic.
	statWriter *dsv.Writer
}

//
// Initialize will start the runtime for processing by saving start time and
// opening stats file.
//
func (runtime *Runtime) Initialize() error {
	runtime.statTotal.StartTime = time.Now().Unix()

	return runtime.OpenStatsFile()
}

//
// Finalize finish the runtime, compute total statistic, write it to file, and
// close the file.
//
func (runtime *Runtime) Finalize() (e error) {
	st := &runtime.statTotal

	st.EndTime = time.Now().Unix()
	st.ElapsedTime = st.EndTime - st.StartTime
	st.Id = int64(len(runtime.stats))

	e = runtime.WriteStat(st)
	if e != nil {
		return e
	}

	return runtime.CloseStatsFile()
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
// ComputeStatFromCM will compute statistic using confusion matrix.
//
func (runtime *Runtime) ComputeStatFromCM(stat *Stat, cm *ConfusionMatrix) {

	stat.OobError = cm.GetFalseRate()

	stat.OobErrorMean = runtime.statTotal.OobError /
		float64(len(runtime.stats)+1)

	stat.TP = int64(cm.TP())
	stat.FP = int64(cm.FP())
	stat.TN = int64(cm.TN())
	stat.FN = int64(cm.FN())
	stat.TPRate = float64(stat.TP) / float64(stat.TP+stat.FN)
	stat.FPRate = float64(stat.FP) / float64(stat.FP+stat.TN)
	stat.TNRate = float64(stat.TN) / float64(stat.FP+stat.TN)
	stat.Precision = float64(stat.TP) / float64(stat.TP+stat.FP)
	stat.FMeasure = 2 / ((1 / stat.Precision) + (1 / stat.TPRate))
	stat.Accuracy = float64(stat.TP+stat.TN) /
		float64(stat.TP+stat.TN+stat.FP+stat.FN)

	if DEBUG >= 1 {
		runtime.PrintOobStat(stat, cm)
		runtime.PrintStat(stat)
	}
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

//
// PrintOobStat will print the out-of-bag statistic to standard output.
//
func (runtime *Runtime) PrintOobStat(stat *Stat, cm *ConfusionMatrix) {
	fmt.Printf("[classifier.runtime] OOB error rate: %.4f,"+
		" total: %.4f, mean %.4f, true rate: %.4f\n",
		stat.OobError, runtime.statTotal.OobError,
		stat.OobErrorMean, cm.GetTrueRate())
}

//
// PrintStat will print statistic value to standard output.
//
func (runtime *Runtime) PrintStat(stat *Stat) {
	if stat == nil {
		statslen := len(runtime.stats)
		if statslen <= 0 {
			return
		}
		stat = runtime.stats[statslen-1]
	}

	fmt.Printf("[classifier.runtime] TPRate: %.4f, FPRate: %.4f,"+
		" TNRate: %.4f, precision: %.4f, f-measure: %.4f,"+
		" accuracy: %.4f\n", stat.TPRate, stat.FPRate, stat.TNRate,
		stat.Precision, stat.FMeasure, stat.Accuracy)
}

//
// PrintStatTotal will print total statistic to standard output.
//
func (runtime *Runtime) PrintStatTotal(st *Stat) {
	if st == nil {
		st = &runtime.statTotal
	}
	runtime.PrintStat(st)
}
