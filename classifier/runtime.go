// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package classifier

import (
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/numerus"
	"github.com/shuLhan/tabula"
	"github.com/shuLhan/tekstus"
	"math"
	"os"
	"strconv"
	"time"
)

var (
	// RuntimeDebug level, can be set it from environment variable
	// "RuntimeDebug".
	RuntimeDebug = 0
)

//
// Runtime define a generic type which provide common fields that can be
// embedded by the real classifier (e.g. RandomForest).
//
type Runtime struct {
	// RunOOB if its true the OOB will be computed, default is false.
	RunOOB bool `json:"RunOOB"`

	// OOBStatsFile is the file where OOB statistic will be written.
	OOBStatsFile string `json:"OOBStatsFile"`

	// PerfFile is the file where statistic of performance will be written.
	PerfFile string `json:"PerfFile"`

	// oobCms contain confusion matrix value for each OOB in iteration.
	oobCms []CM

	// oobStats contain statistic of classifier for each OOB in iteration.
	oobStats Stats

	// oobStatTotal contain total OOB statistic values.
	oobStatTotal Stat

	// oobWriter contain file writer for statistic.
	oobWriter *dsv.Writer

	// perfs contain performance statistic per sample, after classifying
	// sample on classifier.
	perfs Stats
}

func init() {
	var e error
	RuntimeDebug, e = strconv.Atoi(os.Getenv("RUNTIME_DEBUG"))
	if e != nil {
		RuntimeDebug = 0
	}
}

//
// Initialize will start the runtime for processing by saving start time and
// opening stats file.
//
func (runtime *Runtime) Initialize() error {
	runtime.oobStatTotal.StartTime = time.Now().Unix()

	return runtime.OpenOOBStatsFile()
}

//
// Finalize finish the runtime, compute total statistic, write it to file, and
// close the file.
//
func (runtime *Runtime) Finalize() (e error) {
	st := &runtime.oobStatTotal

	st.EndTime = time.Now().Unix()
	st.ElapsedTime = st.EndTime - st.StartTime
	st.ID = int64(len(runtime.oobStats))

	e = runtime.WriteOOBStat(st)
	if e != nil {
		return e
	}

	return runtime.CloseOOBStatsFile()
}

//
// OOBStats return all statistic objects.
//
func (runtime *Runtime) OOBStats() *Stats {
	return &runtime.oobStats
}

//
// StatTotal return total statistic.
//
func (runtime *Runtime) StatTotal() *Stat {
	return &runtime.oobStatTotal
}

//
// AddOOBCM will append new confusion matrix.
//
func (runtime *Runtime) AddOOBCM(cm *CM) {
	runtime.oobCms = append(runtime.oobCms, *cm)
}

//
// AddStat will append new classifier statistic data.
//
func (runtime *Runtime) AddStat(stat *Stat) {
	runtime.oobStats = append(runtime.oobStats, stat)
}

//
// ComputeCM will compute confusion matrix of sample using value space, actual
// and prediction values.
//
func (runtime *Runtime) ComputeCM(sampleIds []int,
	vs, actuals, predicts []string,
) (
	cm *CM,
) {
	cm = &CM{}

	cm.ComputeStrings(vs, actuals, predicts)
	cm.GroupIndexPredictionsStrings(sampleIds, actuals, predicts)

	if RuntimeDebug >= 2 {
		fmt.Println("[classifier.runtime]", cm)
	}

	return cm
}

//
// ComputeStatFromCM will compute statistic using confusion matrix.
//
func (runtime *Runtime) ComputeStatFromCM(stat *Stat, cm *CM) {

	stat.OobError = cm.GetFalseRate()

	stat.OobErrorMean = runtime.oobStatTotal.OobError /
		float64(len(runtime.oobStats)+1)

	stat.TP = int64(cm.TP())
	stat.FP = int64(cm.FP())
	stat.TN = int64(cm.TN())
	stat.FN = int64(cm.FN())

	t := float64(stat.TP + stat.FN)
	if t == 0 {
		stat.TPRate = 0
	} else {
		stat.TPRate = float64(stat.TP) / t
	}

	t = float64(stat.FP + stat.TN)
	if t == 0 {
		stat.FPRate = 0
	} else {
		stat.FPRate = float64(stat.FP) / t
	}

	t = float64(stat.FP + stat.TN)
	if t == 0 {
		stat.TNRate = 0
	} else {
		stat.TNRate = float64(stat.TN) / t
	}

	t = float64(stat.TP + stat.FP)
	if t == 0 {
		stat.Precision = 0
	} else {
		stat.Precision = float64(stat.TP) / t
	}

	t = (1 / stat.Precision) + (1 / stat.TPRate)
	if t == 0 {
		stat.FMeasure = 0
	} else {
		stat.FMeasure = 2 / t
	}

	t = float64(stat.TP + stat.TN + stat.FP + stat.FN)
	if t == 0 {
		stat.Accuracy = 0
	} else {
		stat.Accuracy = float64(stat.TP+stat.TN) / t
	}

	if RuntimeDebug >= 1 {
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

	nstat := len(runtime.oobStats)
	if nstat == 0 {
		return
	}

	t := &runtime.oobStatTotal

	t.OobError += stat.OobError
	t.OobErrorMean = t.OobError / float64(nstat)
	t.TP += stat.TP
	t.FP += stat.FP
	t.TN += stat.TN
	t.FN += stat.FN

	total := float64(t.TP + t.FN)
	if total == 0 {
		t.TPRate = 0
	} else {
		t.TPRate = float64(t.TP) / total
	}

	total = float64(t.FP + t.TN)
	if total == 0 {
		t.FPRate = 0
	} else {
		t.FPRate = float64(t.FP) / total
	}

	total = float64(t.FP + t.TN)
	if total == 0 {
		t.TNRate = 0
	} else {
		t.TNRate = float64(t.TN) / total
	}

	total = float64(t.TP + t.FP)
	if total == 0 {
		t.Precision = 0
	} else {
		t.Precision = float64(t.TP) / total
	}

	total = (1 / t.Precision) + (1 / t.TPRate)
	if total == 0 {
		t.FMeasure = 0
	} else {
		t.FMeasure = 2 / total
	}

	total = float64(t.TP + t.TN + t.FP + t.FN)
	if total == 0 {
		t.Accuracy = 0
	} else {
		t.Accuracy = float64(t.TP+t.TN) / total
	}
}

//
// OpenOOBStatsFile will open statistic file for output.
//
func (runtime *Runtime) OpenOOBStatsFile() error {
	if runtime.oobWriter != nil {
		_ = runtime.CloseOOBStatsFile()
	}
	runtime.oobWriter = &dsv.Writer{}
	return runtime.oobWriter.OpenOutput(runtime.OOBStatsFile)
}

//
// WriteOOBStat will write statistic of process to file.
//
func (runtime *Runtime) WriteOOBStat(stat *Stat) error {
	if runtime.oobWriter == nil {
		return nil
	}
	if stat == nil {
		return nil
	}
	return runtime.oobWriter.WriteRawRow(stat.ToRow(), nil, nil)
}

//
// CloseOOBStatsFile will close statistics file for writing.
//
func (runtime *Runtime) CloseOOBStatsFile() (e error) {
	if runtime.oobWriter == nil {
		return
	}

	e = runtime.oobWriter.Close()
	runtime.oobWriter = nil

	return
}

//
// PrintOobStat will print the out-of-bag statistic to standard output.
//
func (runtime *Runtime) PrintOobStat(stat *Stat, cm *CM) {
	fmt.Printf("[classifier.runtime] OOB error rate: %.4f,"+
		" total: %.4f, mean %.4f, true rate: %.4f\n",
		stat.OobError, runtime.oobStatTotal.OobError,
		stat.OobErrorMean, cm.GetTrueRate())
}

//
// PrintStat will print statistic value to standard output.
//
func (runtime *Runtime) PrintStat(stat *Stat) {
	if stat == nil {
		statslen := len(runtime.oobStats)
		if statslen <= 0 {
			return
		}
		stat = runtime.oobStats[statslen-1]
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
		st = &runtime.oobStatTotal
	}
	runtime.PrintStat(st)
}

//
// Performance given an actuals class label and their probabilities, compute
// the performance statistic of classifier.
//
// Algorithm,
// (1) Sort the probabilities in descending order.
// (2) Sort the actuals and predicts using sorted index from probs
// (3) Compute tpr, fpr, precision
// (4) Write performance to file.
//
func (rt *Runtime) Performance(samples tabula.ClasetInterface,
	predicts []string, probs []float64,
) (
	perfs Stats,
) {
	// (1)
	actuals := samples.GetClassAsStrings()
	sortedIds := numerus.IntCreateSeq(0, len(probs)-1)
	numerus.Floats64InplaceMergesort(probs, sortedIds, 0, len(probs),
		false)

	// (2)
	tekstus.StringsSortByIndex(&actuals, sortedIds)
	tekstus.StringsSortByIndex(&predicts, sortedIds)

	// (3)
	rt.computePerfByProbs(samples, actuals, probs)

	return rt.perfs
}

//
// computePerfByProbs will compute classifier performance using probabilities
// or score `probs`.
//
// This currently only work for two class problem.
//
func (rt *Runtime) computePerfByProbs(samples tabula.ClasetInterface,
	actuals []string, probs []float64,
) {
	vs := samples.GetClassValueSpace()
	nactuals := numerus.IntsTo64(samples.Counts())
	pprev := math.Inf(-1)
	tp := int64(0)
	fp := int64(0)

	for x, p := range probs {
		if p != pprev {
			stat := Stat{}
			stat.SetTPRate(tp, nactuals[0])
			stat.SetFPRate(fp, nactuals[1])
			stat.SetPrecisionFromRate(nactuals[0], nactuals[1])

			rt.perfs = append(rt.perfs, &stat)
			pprev = p
		}

		if actuals[x] == vs[0] {
			tp++
		} else {
			fp++
		}
	}

	stat := Stat{}
	stat.SetTPRate(tp, nactuals[0])
	stat.SetFPRate(fp, nactuals[1])
	stat.SetPrecisionFromRate(nactuals[0], nactuals[1])

	rt.perfs = append(rt.perfs, &stat)

	if len(rt.perfs) >= 2 {
		// Replace the first stat with second stat, because of NaN
		// value on the first precision.
		rt.perfs[0] = rt.perfs[1]
	}
}

//
// WritePerformance will write performance data to file.
//
func (rt *Runtime) WritePerformance() error {
	return rt.perfs.Write(rt.PerfFile)
}
