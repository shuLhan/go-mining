// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package randomforest implement ensemble of classifiers using random forest
algorithm by Breiman and Cutler.

	Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.

The implementation is based on various sources and using author experience.
*/
package randomforest

import (
	"errors"
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/classifiers"
	"github.com/shuLhan/go-mining/classifiers/cart"
	"github.com/shuLhan/tabula"
	"github.com/shuLhan/tabula/util"
	"github.com/shuLhan/tekstus"
	"math"
	"os"
	"strconv"
	"time"
)

const (
	// DefNumTree default number of tree.
	DefNumTree = 100
	// DefPercentBoot default percentage of sample that will be used for
	// bootstraping a tree.
	DefPercentBoot = 66
	// DefStatsFile default statistic file output.
	DefStatsFile = "randomforest.stats.dat"
)

var (
	// DEBUG level, set it from environment variable.
	DEBUG = 0
)

var (
	// ErrNoInput will tell you when no input is given.
	ErrNoInput = errors.New("randomforest: input reader is empty")
)

/*
Runtime contains input and output configuration when generating random forest.
*/
type Runtime struct {
	// NTree number of tree in forest.
	NTree int `json:"NTree"`
	// NRandomFeature number of feature randomly selected for each tree.
	NRandomFeature int `json:"NRandomFeature"`
	// PercentBoot percentage of sample for bootstraping.
	PercentBoot int `json:"PercentBoot"`
	// StatsFile is the file where performance statistic will be written.
	StatsFile string `json:"StatsFile"`

	// nSubsample number of samples used for bootstraping.
	nSubsample int
	// trees contain all tree in the forest.
	trees []cart.Runtime
	// bagIndices contain list of index of selected samples at bootstraping
	// for book-keeping.
	bagIndices [][]int

	// oobErrorTotal contain the total of out-of-bag error value.
	oobErrorTotal float64
	// oobErrorSteps contain OOB error for each steps building a forest
	// from the first tree until `n` tree.
	oobErrorSteps []float64
	// oobErrorStepsMean contain mean for OOB error for each steps, using
	// oobErrorTotal / current-number-of-tree
	oobErrorStepsMean []float64

	// cmatrices contain confusion matrix from the first tree until NTree.
	cmatrices []classifiers.ConfusionMatrix
	// stats contain statistic of classifier for each tree
	stats classifiers.Stats
}

func init() {
	v := os.Getenv("RANDOMFOREST_DEBUG")
	if v == "" {
		DEBUG = 0
	} else {
		DEBUG, _ = strconv.Atoi(v)
	}
}

/*
New check and initialize forest input and attributes.
`ntree` is number of tree to be generated.
`nfeature` is number of feature randomly selected for each tree for splitting
(the `m`).
`percentboot` is percentage of sample that will be taken randomly for
generating a tree.
*/
func New(ntree, nfeature, percentboot int) (forest *Runtime) {
	if ntree <= 0 {
		ntree = DefNumTree
	}
	if percentboot <= 0 {
		percentboot = DefPercentBoot
	}

	forest = &Runtime{
		NTree:          ntree,
		NRandomFeature: nfeature,
		PercentBoot:    percentboot,
	}

	return forest
}

/*
Trees return all tree in forest.
*/
func (forest *Runtime) Trees() []cart.Runtime {
	return forest.trees
}

/*
AddCartTree add tree to forest
*/
func (forest *Runtime) AddCartTree(tree cart.Runtime) {
	forest.trees = append(forest.trees, tree)
}

/*
AddBagIndex add bagging index for book keeping.
*/
func (forest *Runtime) AddBagIndex(bagIndex []int) {
	forest.bagIndices = append(forest.bagIndices, bagIndex)
}

/*
AddConfusionMatrix will append new confusion matrix.
*/
func (forest *Runtime) AddConfusionMatrix(cm *classifiers.ConfusionMatrix) {
	forest.cmatrices = append(forest.cmatrices, *cm)
}

/*
AddStat will append new classifier statistic data.
*/
func (forest *Runtime) AddStat(stat *classifiers.Stat) {
	forest.stats = append(forest.stats, *stat)
}

/*
OobErrorTotal return total of OOB error.
*/
func (forest *Runtime) OobErrorTotal() float64 {
	return forest.oobErrorTotal
}

/*
OobErrorTotalMean return mean of all OOB errors.
*/
func (forest *Runtime) OobErrorTotalMean() float64 {
	return forest.oobErrorTotal / float64(forest.NTree)
}

/*
OobErrorSteps return all OOB error in each step when building forest.
*/
func (forest *Runtime) OobErrorSteps() []float64 {
	return forest.oobErrorSteps
}

/*
OobErrorStepsMean return the last error mean value.
*/
func (forest *Runtime) OobErrorStepsMean() []float64 {
	return forest.oobErrorStepsMean
}

//
// Stats return forest statistics for all tree.
//
func (forest *Runtime) Stats() *classifiers.Stats {
	return &forest.stats
}

/*
Init will check forest inputs and set it to default values if invalid.
*/
func (forest *Runtime) Init(samples tabula.ClasetInterface) {
	if forest.NTree <= 0 {
		forest.NTree = DefNumTree
	}
	if forest.PercentBoot <= 0 {
		forest.PercentBoot = DefPercentBoot
	}
	if forest.NRandomFeature <= 0 {
		// Set default value to square-root of features.
		ncol := samples.GetNColumn() - 1
		forest.NRandomFeature = int(math.Sqrt(float64(ncol)))
	}
	if forest.StatsFile == "" {
		forest.StatsFile = DefStatsFile
	}
}

/*
Build the forest using samples dataset.

Algorithm,

(0) Recheck input value: number of tree, percentage bootstrap.
(1) Calculate number of random samples for each tree.

	number-of-sample * percentage-of-bootstrap

(2) Grow tree in forest
(2.1) Create new tree, repeat until success.
(3) Write performance statistic to file.
*/
func (forest *Runtime) Build(samples tabula.ClasetInterface) (e error) {
	// check input samples
	if samples == nil {
		return ErrNoInput
	}

	// (0)
	forest.Init(samples)

	// (1)
	forest.nSubsample = int(float32(samples.GetNRow()) *
		(float32(forest.PercentBoot) / 100.0))

	if DEBUG >= 1 {
		fmt.Println("[randomforest] forest:", forest)
	}

	// (2)
	for t := 0; t < forest.NTree; t++ {
		if DEBUG >= 1 {
			fmt.Printf("----\n[randomforest] tree # %d\n", t)
		}

		// (2.1)
		for {
			_, _, e = forest.GrowTree(samples)
			if e == nil {
				break
			}
		}
	}

	// (3)
	return forest.writeStats()
}

/*
GrowTree build a new tree in forest, return OOB error value or error if tree
can not grow.

Algorithm,

(1) Select random samples with replacement, also with OOB.
(2) Build tree using CART, without pruning.
(3) Add tree to forest.
(4) Save index of random samples for calculating error rate later.
(5) Run OOB on forest.
(6) Calculate OOB error rate and statistic values.
*/
func (forest *Runtime) GrowTree(samples tabula.ClasetInterface) (
	cm *classifiers.ConfusionMatrix, stat *classifiers.Stat, e error,
) {
	start := time.Now()

	// (1)
	bag, oob, bagIdx, oobIdx := tabula.RandomPickRows(
		samples.(tabula.DatasetInterface),
		forest.nSubsample, true)

	bagset := bag.(tabula.ClasetInterface)

	if DEBUG >= 2 {
		bagset.RecountMajorMinor()
		fmt.Println("[randomforest] Bagging:", bagset)
	}

	// (2)
	cart, e := cart.New(bagset, cart.SplitMethodGini,
		forest.NRandomFeature)
	if e != nil {
		return nil, nil, e
	}

	// (3)
	forest.AddCartTree(*cart)

	// (4)
	forest.AddBagIndex(bagIdx)

	// (5)
	oobset := oob.(tabula.ClasetInterface)
	cm = forest.ClassifySet(oobset, oobIdx, true)
	forest.AddConfusionMatrix(cm)

	// (6)
	stat = forest.computeStatistic(cm)

	stat.EndTime = time.Now()
	stat.StartTime = start

	forest.AddStat(stat)

	if DEBUG >= 1 {
		fmt.Printf("[randomforest] Elapsed time: %s\n",
			stat.EndTime.Sub(stat.StartTime))
	}

	return cm, stat, nil
}

/*
ClassifySet given a dataset predict their class by running each sample in
forest. Return miss classification rate:

	(number of missed class / number of samples).

Algorithm,

(0) Get value space (possible class values in dataset)
(1) Save test-set target values.
(2) Clear target values in test-set.
(3) For each row in test-set,
(3.1) for each tree in forest,
(3.1.1) If row is used to build the tree then skip it,
(3.1.2) classify row in tree,
(3.1.3) save tree class value.
(3.2) Collect majority class vote in forest.
(4) Compute confusion matrix from predictions.
(5) Restore original target values in testset.
*/
func (forest *Runtime) ClassifySet(testset tabula.ClasetInterface,
	testsetIdx []int, uniq bool,
) (
	cm *classifiers.ConfusionMatrix,
) {
	// (0)
	targetVS := testset.GetClassValueSpace()

	// (1)
	targetAttr := testset.GetClassColumn()
	targetValues := targetAttr.ToStringSlice()
	indexlen := len(testsetIdx)

	// (2)
	targetAttr.ClearValues()

	// (3)
	rows := testset.GetRows()
	for x, row := range *rows {
		var votes []string

		// (3.1)
		for y, tree := range forest.trees {
			// (3.1.1)
			if uniq && indexlen > 0 {
				exist := util.IntIsExist(forest.bagIndices[y],
					testsetIdx[x])
				if exist {
					continue
				}
			}

			// (3.1.2)
			class := tree.Classify(row)

			// (3.1.3)
			votes = append(votes, class)
		}

		// (3.2)
		class := tekstus.WordsMaxCountOf(votes, targetVS, false)
		(*targetAttr).Records[x].V = class
	}

	// (4)
	predictions := targetAttr.ToStringSlice()

	cm = classifiers.NewConfusionMatrix(targetVS, targetValues,
		predictions)

	if DEBUG >= 2 {
		fmt.Println("[randomforest]", cm)
	}

	// (5)
	targetAttr.SetValues(targetValues)

	return cm
}

//
// computeStatistic of random forest using confusion matrix and return it.
//
func (forest *Runtime) computeStatistic(cm *classifiers.ConfusionMatrix) (
	stat *classifiers.Stat,
) {
	oobErr := cm.GetFalseRate()

	forest.oobErrorSteps = append(forest.oobErrorSteps, oobErr)
	forest.oobErrorTotal += oobErr

	oobErrTotalMean := forest.oobErrorTotal /
		float64(len(forest.oobErrorSteps))
	forest.oobErrorStepsMean = append(forest.oobErrorStepsMean,
		oobErrTotalMean)

	stat = &classifiers.Stat{}

	tp := cm.TP()
	fp := cm.FP()
	fn := cm.FN()
	tn := cm.TN()
	stat.TPRate = float64(tp) / float64(tp+fn)
	stat.FPRate = float64(fp) / float64(fp+tn)
	stat.Precision = float64(tp) / float64(tp+fp)
	stat.FMeasure = 2 / ((1 / stat.Precision) + (1 / stat.TPRate))
	stat.Accuracy = float64(tp+tn) / float64(tp+tn+fp+fn)

	if DEBUG >= 1 {
		fmt.Printf("[randomforest] OOB error rate: %.4f,"+
			" total: %.4f, mean %.4f, true rate: %.4f\n",
			cm.GetFalseRate(), forest.oobErrorTotal,
			forest.OobErrorTotalMean(), cm.GetTrueRate())

		fmt.Printf("[randomforest] TPRate: %.4f, FPRate: %.4f,"+
			" precision: %.4f, f-measure: %.4f, accuracy: %.4f\n",
			stat.TPRate, stat.FPRate, stat.Precision,
			stat.FMeasure, stat.Accuracy)
	}

	return
}

//
// writeStats will write performance statistic to file in CSV format.
// The file will contain,
// - tree number
// - start timestamp
// - end timestamp
// - elapsed time in seconds
// - TP
// - FP
// - TN
// - FN
// - OOB error in current tree
// - OOB error mean in current tree
// - TP rate
// - FP rate
// - Precision
// - F-measure
// - Accuracy
//
// and total of all of them.
//
func (forest *Runtime) writeStats() (e error) {
	stats := forest.getAllStats()

	writer, e := dsv.NewWriter("")
	if e != nil {
		fmt.Println("[randomforest] error: ", e)
		return e
	}

	e = writer.OpenOutput(forest.StatsFile)
	if e != nil {
		fmt.Println("[randomforest] error: ", e)
		return e
	}

	sep := ","
	n, e := writer.WriteRawDataset(stats, &sep)
	if e != nil {
		fmt.Println("[randomforest] error: ", e)
		return e
	}

	exp := stats.Len()
	if exp != n {
		fmt.Println("[randomforest] expecting", exp, " rows, got ", n)
		return e
	}

	e = writer.Close()
	if e != nil {
		fmt.Println("[randomforest] error: ", e)
		return e
	}

	return nil
}

//
// getAllStats return all statistic value and their total/mean values.
//
func (forest *Runtime) getAllStats() (stats *tabula.Dataset) {
	stats = tabula.NewDataset(tabula.DatasetModeMatrix, nil, nil)

	treeIds := util.IntCreateSequence(1, int64(forest.NTree))
	col := tabula.NewColumnInt(treeIds, "tree_id")
	stats.PushColumn(*col)

	startTimes := forest.stats.StartTimes()
	col = tabula.NewColumnInt(startTimes, "start_time")
	stats.PushColumn(*col)

	endTimes := forest.stats.EndTimes()
	col = tabula.NewColumnInt(endTimes, "end_time")
	stats.PushColumn(*col)

	elapsedTimes := classifiers.ComputeElapsedTimes(startTimes, endTimes)
	col = tabula.NewColumnInt(elapsedTimes, "elapsed_time")
	stats.PushColumn(*col)

	tp, fp, tn, fn := forest.GetConfusionMatrixValues()
	col = tabula.NewColumnInt(tp, "tp")
	stats.PushColumn(*col)
	col = tabula.NewColumnInt(fp, "fp")
	stats.PushColumn(*col)
	col = tabula.NewColumnInt(tn, "tn")
	stats.PushColumn(*col)
	col = tabula.NewColumnInt(fn, "fn")
	stats.PushColumn(*col)

	col = tabula.NewColumnReal(forest.oobErrorSteps, "oob_error")
	stats.PushColumn(*col)

	col = tabula.NewColumnReal(forest.oobErrorStepsMean, "oob_error_mean")
	stats.PushColumn(*col)

	tprates := forest.stats.TPRates()
	col = tabula.NewColumnReal(tprates, "tprate")
	stats.PushColumn(*col)

	fprates := forest.stats.FPRates()
	col = tabula.NewColumnReal(fprates, "fprate")
	stats.PushColumn(*col)

	precisions := forest.stats.Precisions()
	col = tabula.NewColumnReal(precisions, "precision")
	stats.PushColumn(*col)

	fmeasures := forest.stats.FMeasures()
	col = tabula.NewColumnReal(fmeasures, "fmeasure")
	stats.PushColumn(*col)

	accuracies := forest.stats.Accuracies()
	col = tabula.NewColumnReal(accuracies, "accuracy")
	stats.PushColumn(*col)

	//
	// Compute total and mean of all statistic values.
	//
	// - Total tree in forest.
	total := tabula.Row{}
	ntree := float64(stats.Len())
	total.PushBack(tabula.NewRecordInt(int64(stats.Len())))

	// - Start time of the first tree.
	starttime := time.Now().Unix()
	if len(startTimes) > 0 {
		starttime = startTimes[0]
	}
	total.PushBack(tabula.NewRecordInt(starttime))

	// - End time of the last tree.
	endTimesLen := len(endTimes)
	endtime := time.Now().Unix()
	if endTimesLen > 0 {
		endtime = endTimes[endTimesLen-1]
	}
	total.PushBack(tabula.NewRecordInt(endtime))

	// - Total elapsed time.
	total.PushBack(tabula.NewRecordInt(endtime - starttime))

	// - Total TP.
	total.PushBack(tabula.NewRecordInt(tekstus.Int64Sum(tp)))
	// - Total FP.
	total.PushBack(tabula.NewRecordInt(tekstus.Int64Sum(fp)))
	// - Total TN.
	total.PushBack(tabula.NewRecordInt(tekstus.Int64Sum(tn)))
	// - Total FN.
	total.PushBack(tabula.NewRecordInt(tekstus.Int64Sum(fn)))

	// - Mean of OOB error in all steps.
	totalv := tekstus.Float64Sum(forest.oobErrorSteps) / ntree
	total.PushBack(tabula.NewRecordReal(totalv))

	// - Mean OOB error mean in all steps.
	totalv = tekstus.Float64Sum(forest.oobErrorStepsMean) / ntree
	total.PushBack(tabula.NewRecordReal(totalv))

	// - Mean of tp-rates.
	totalv = tekstus.Float64Sum(tprates) / ntree
	total.PushBack(tabula.NewRecordReal(totalv))

	// - Mean of fp-rates.
	totalv = tekstus.Float64Sum(fprates) / ntree
	total.PushBack(tabula.NewRecordReal(totalv))

	// - Mean of precisions.
	totalv = tekstus.Float64Sum(precisions) / ntree
	total.PushBack(tabula.NewRecordReal(totalv))

	// - Mean of F-measures.
	totalv = tekstus.Float64Sum(fmeasures) / ntree
	total.PushBack(tabula.NewRecordReal(totalv))

	// - Mean of accuracies.
	totalv = tekstus.Float64Sum(accuracies) / ntree
	total.PushBack(tabula.NewRecordReal(totalv))

	if DEBUG >= 1 {
		fmt.Println("[randomforest] Total:", total)
	}

	stats.PushRow(total)

	return stats
}

//
// GetConfusionMatrixValues return all true positive, false positive, true
// negative, and false negative values in confusion matrix.
//
func (forest *Runtime) GetConfusionMatrixValues() (tp, fp, tn, fn []int64) {
	for _, cm := range forest.cmatrices {
		tp = append(tp, int64(cm.TP()))
		fp = append(fp, int64(cm.FP()))
		tn = append(tn, int64(cm.TN()))
		fn = append(fn, int64(cm.FN()))
	}
	return
}
