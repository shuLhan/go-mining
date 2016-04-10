// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package cascadedrf implement the cascaded random forest algorithm, proposed by
Baumann et.al in their paper:

	Baumann, Florian, et al. "Cascaded Random Forest for Fast Object
	Detection." Image Analysis. Springer Berlin Heidelberg, 2013. 131-142.

*/
package cascadedrf

import (
	"errors"
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/classifiers"
	"github.com/shuLhan/go-mining/classifiers/randomforest"
	"github.com/shuLhan/tabula"
	"math"
	"os"
	"strconv"
	"time"
)

const (
	// DefStage default number of stage
	DefStage = 200
	// DefTPRate default threshold for true-positive rate.
	DefTPRate = 0.9
	// DefTNRate default threshold for true-negative rate.
	DefTNRate = 0.7

	// DefNumTree default number of tree.
	DefNumTree = 1
	// DefPercentBoot default percentage of sample that will be used for
	// bootstraping a tree.
	DefPercentBoot = 66
	// DefStatsFile default statistic file output.
	DefStatsFile = "cascadedrf.stats"
)

var (
	// DEBUG level, can set from environment CRF_DEBUG variable.
	DEBUG = 0
)

var (
	// ErrNoInput will tell you when no input is given.
	ErrNoInput = errors.New("randomforest: input samples is empty")
)

/*
Runtime define the cascaded random forest runtime input and output.
*/
type Runtime struct {
	// NStage number of stage.
	NStage int `json:"NStage"`
	// TPRate threshold for true positive rate per stage.
	TPRate float64 `json:"TPRate"`
	// TNRate threshold for true negative rate per stage.
	TNRate float64 `json:"TNRate"`

	// NTree number of tree in each stage.
	NTree int `json:"NTree"`
	// NRandomFeature number of features used to split the dataset.
	NRandomFeature int
	// PercentBoot percentage of bootstrap.
	PercentBoot int `json:"PercentBoot"`
	// StatsFile is the file where performance statistic will be written.
	StatsFile string `json:"StatsFile"`

	// Stages contain list of cascaded stages.
	stages []Stage

	// cmatrices contain confusion matrix from the first stage until
	// NStage.
	cmatrices []classifiers.ConfusionMatrix
	// stats contain statistic of classifier for each stage.
	stats classifiers.Stats
	// StatTotal contain total statistic values.
	statTotal classifiers.Stat
	// statWriter contain file writer for statistic.
	statWriter *dsv.Writer
}

func init() {
	var e error
	DEBUG, e = strconv.Atoi(os.Getenv("CRF_DEBUG"))
	if e != nil {
		DEBUG = 0
	}
}

/*
New create and return new input for cascaded random-forest.
*/
func New(nstage, ntree, percentboot, nfeature int,
	tprate, tnrate float64,
	samples tabula.ClasetInterface,
) (
	crf *Runtime,
) {
	crf = &Runtime{
		NStage:         nstage,
		NTree:          ntree,
		PercentBoot:    percentboot,
		NRandomFeature: nfeature,
		TPRate:         tprate,
		TNRate:         tnrate,
	}

	crf.Init(samples)

	return crf
}

//
// Init will check crf inputs and set it to default values if invalid.
//
func (crf *Runtime) Init(samples tabula.ClasetInterface) {
	if crf.NStage <= 0 {
		crf.NStage = DefStage
	}
	if crf.TPRate <= 0 || crf.TPRate >= 1 {
		crf.TPRate = DefTPRate
	}
	if crf.TNRate <= 0 || crf.TNRate >= 1 {
		crf.TNRate = DefTNRate
	}
	if crf.NTree <= 0 {
		crf.NTree = DefNumTree
	}
	if crf.PercentBoot <= 0 {
		crf.PercentBoot = DefPercentBoot
	}
	if crf.NRandomFeature <= 0 {
		// Set default value to square-root of features.
		ncol := samples.GetNColumn() - 1
		crf.NRandomFeature = int(math.Sqrt(float64(ncol)))
	}
	if crf.StatsFile == "" {
		crf.StatsFile = DefStatsFile
	}

	crf.statTotal.Id = int64(crf.NStage)
}

//
// Build given a sample dataset, build the stage and randomforest.
//
// Algorithm,
// (1) For 0 to maximum of stage,
// (1.1) grow a forest in current stage.
//
func (crf *Runtime) Build(samples tabula.ClasetInterface) (e error) {
	if samples == nil {
		return ErrNoInput
	}

	if DEBUG >= 1 {
		fmt.Println("[cascadedrf] # samples:", samples.Len())
		fmt.Println("[cascadedrf] sample:", samples.GetRow(0))
	}

	crf.Init(samples)

	e = crf.openStatsFile()
	if e != nil {
		fmt.Println("[cascadedrf] error: ", e)
		return
	}

	if DEBUG >= 1 {
		fmt.Println("[cascadedrf]", crf)
	}

	crf.statTotal.StartTime = time.Now().Unix()

	// (1)
	for x := 0; x < crf.NStage; x++ {
		if DEBUG >= 1 {
			fmt.Printf("====\n[cascadedrf] stage # %d\n", x)
		}

		// (1.1)
		_ = crf.createForest(samples)
	}

	crf.statTotal.EndTime = time.Now().Unix()
	crf.statTotal.ElapsedTime = crf.statTotal.EndTime -
		crf.statTotal.StartTime

	crf.statTotal.Id = int64(len(crf.stats))
	e = crf.writeStat(&crf.statTotal)

	return e
}

//
// createForest will create and return a forest and run the training `samples`
// on it.
//
// Algorithm,
// (1) Initialize forest.
// (2) For 0 to maximum number of tree in forest,
// (2.1) grow one tree until success.
// (2.2) If tree tp-rate and tn-rate greater than threshold, stop growing.
// (3) Compute and write forest statistic to file.
//
func (crf *Runtime) createForest(samples tabula.ClasetInterface) (
	forest *randomforest.Runtime,
) {
	var stat *classifiers.Stat
	var e error

	// (1)
	forest = randomforest.New(crf.NTree, crf.NRandomFeature,
		crf.PercentBoot, samples)

	ft := forest.StatTotal()
	ft.StartTime = time.Now().Unix()

	// (2)
	for t := 0; t < crf.NTree; t++ {
		if DEBUG >= 1 {
			fmt.Printf("[cascadedrf] tree # %d\n", t)
		}

		// (2.1)
		for {
			_, stat, e = forest.GrowTree(samples)
			if e == nil {
				break
			}
		}

		if DEBUG >= 1 {
			fmt.Printf("[cascadedrf] forest stat: "+
				" TPRate %.4f, FPRate %.4f, TNRate %.4f,"+
				" Precision %.4f, FMeasure %.4f,"+
				" Accuracy %.4f\n",
				ft.TPRate, ft.FPRate, ft.TNRate,
				ft.Precision, ft.FMeasure,
				ft.Accuracy)
		}

		// (2.2)
		if stat.TPRate > crf.TPRate &&
			stat.TNRate > crf.TNRate {
			break
		}
	}

	// (3)
	ft.EndTime = time.Now().Unix()
	ft.ElapsedTime = ft.EndTime - ft.StartTime
	ft.Id = int64(len(crf.stats))

	e = crf.writeStat(ft)
	if e != nil {
		return
	}

	crf.stats.Add(ft)
	crf.computeStatTotal(ft)

	return
}

//
// computeStatTotal compute total statistic.
//
func (crf *Runtime) computeStatTotal(stat *classifiers.Stat) {
	if stat == nil {
		return
	}

	nstat := len(crf.stats)
	if nstat == 0 {
		return
	}

	t := &crf.statTotal

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
// openStatsFile will open statistic file for output.
//
func (crf *Runtime) openStatsFile() error {
	if crf.statWriter != nil {
		_ = crf.closeStatsFile()
	}
	crf.statWriter = &dsv.Writer{}
	return crf.statWriter.OpenOutput(crf.StatsFile)
}

//
// writeStat will write statistic of process to file.
//
func (crf *Runtime) writeStat(stat *classifiers.Stat) error {
	if crf.statWriter == nil {
		return nil
	}
	if stat == nil {
		return nil
	}
	return crf.statWriter.WriteRawRow(stat.ToRow(), nil, nil)
}

//
// closeStatsFile will close statistics file for writing.
//
func (crf *Runtime) closeStatsFile() (e error) {
	if crf.statWriter == nil {
		return
	}

	e = crf.statWriter.Close()
	crf.statWriter = nil

	return
}
