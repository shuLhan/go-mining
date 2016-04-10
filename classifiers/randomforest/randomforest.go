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
	DefStatsFile = "randomforest.stats"
)

var (
	// DEBUG level, set it from environment variable.
	DEBUG = 0
)

var (
	// ErrNoInput will tell you when no input is given.
	ErrNoInput = errors.New("randomforest: input samples is empty")
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

	// cmatrices contain confusion matrix from the first tree until NTree.
	cmatrices []classifiers.ConfusionMatrix
	// stats contain statistic of classifier for each tree.
	stats classifiers.Stats
	// StatTotal contain total statistic values.
	statTotal classifiers.Stat
	// statWriter contain file writer for statistic.
	statWriter *dsv.Writer
}

func init() {
	var e error
	DEBUG, e = strconv.Atoi(os.Getenv("RANDOMFOREST_DEBUG"))
	if e != nil {
		DEBUG = 0
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
func New(ntree, nfeature, percentboot int, samples tabula.ClasetInterface) (
	forest *Runtime,
) {
	forest = &Runtime{
		NTree:          ntree,
		NRandomFeature: nfeature,
		PercentBoot:    percentboot,
	}

	forest.Init(samples)

	return
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
	forest.stats = append(forest.stats, stat)
}

//
// StatTotal will return total statistic.
//
func (forest *Runtime) StatTotal() *classifiers.Stat {
	return &forest.statTotal
}

//
// Stats return forest statistics for all tree.
//
func (forest *Runtime) Stats() *classifiers.Stats {
	return &forest.stats
}

//
// Init will check forest inputs and set it to default values if invalid.
//
// (1) Calculate number of random samples for each tree.
//
//	number-of-sample * percentage-of-bootstrap
//
// (2) Set id of total statistic (equal to number of tree).
//
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

	// (1)
	forest.nSubsample = int(float32(samples.GetNRow()) *
		(float32(forest.PercentBoot) / 100.0))

	// (3)
	forest.statTotal.Id = int64(forest.NTree)
}

/*
Build the forest using samples dataset.

Algorithm,

(0) Recheck input value: number of tree, percentage bootstrap, etc.
(1) Open statistic file output.
(2) Grow tree in forest
(2.1) Create new tree, repeat until all tress has been build.
(3) Compute and write total statistic.
*/
func (forest *Runtime) Build(samples tabula.ClasetInterface) (e error) {
	// check input samples
	if samples == nil {
		return ErrNoInput
	}

	// (0)
	forest.Init(samples)

	// (1)
	e = forest.openStatsFile()
	if e != nil {
		fmt.Println("[randomforest] error: ", e)
		return
	}

	if DEBUG >= 1 {
		fmt.Println("[randomforest] forest:", forest)
	}

	// (2)
	forest.statTotal.StartTime = time.Now().Unix()

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

			fmt.Println("[randomforest] error: ", e)
		}
	}

	// (3)
	forest.statTotal.EndTime = time.Now().Unix()
	forest.statTotal.ElapsedTime = forest.statTotal.EndTime -
		forest.statTotal.StartTime

	e = forest.writeStat(&forest.statTotal)
	if e != nil {
		fmt.Println("[randomforest] error: ", e)
	}

	e = forest.closeStatsFile()
	if e != nil {
		fmt.Println("[randomforest] error: ", e)
	}

	return e
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
	stat = &classifiers.Stat{}
	stat.Id = int64(len(forest.trees))
	stat.StartTime = time.Now().Unix()

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

	stat.EndTime = time.Now().Unix()
	stat.ElapsedTime = stat.EndTime - stat.StartTime

	if DEBUG >= 1 {
		fmt.Printf("[randomforest] Elapsed time: %d s\n",
			stat.ElapsedTime)
	}

	// (6)
	forest.computeStatistic(stat, cm)
	forest.AddStat(stat)
	forest.computeStatTotal(stat)

	e = forest.writeStat(stat)

	return cm, stat, e
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
func (forest *Runtime) computeStatistic(stat *classifiers.Stat,
	cm *classifiers.ConfusionMatrix) {

	stat.OobError = cm.GetFalseRate()

	stat.OobErrorMean = forest.statTotal.OobError /
		float64(len(forest.trees))

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
		fmt.Printf("[randomforest] OOB error rate: %.4f,"+
			" total: %.4f, mean %.4f, true rate: %.4f\n",
			stat.OobError, forest.statTotal.OobError,
			stat.OobErrorMean, cm.GetTrueRate())

		fmt.Printf("[randomforest] TPRate: %.4f, FPRate: %.4f,"+
			" TNRate: %.4f,"+
			" precision: %.4f, f-measure: %.4f, accuracy: %.4f\n",
			stat.TPRate, stat.FPRate, stat.TNRate, stat.Precision,
			stat.FMeasure, stat.Accuracy)
	}
}

//
// computeStatTotal compute total statistic.
//
func (forest *Runtime) computeStatTotal(stat *classifiers.Stat) {
	if stat == nil {
		return
	}

	nstat := len(forest.stats)
	if nstat == 0 {
		return
	}

	t := &forest.statTotal

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
func (forest *Runtime) openStatsFile() error {
	if forest.statWriter != nil {
		_ = forest.closeStatsFile()
	}
	forest.statWriter = &dsv.Writer{}
	return forest.statWriter.OpenOutput(forest.StatsFile)
}

//
// writeStats will write performance statistic to file in CSV format.
// The file will contain,
// - tree number
// - start timestamp
// - end timestamp
// - elapsed time in seconds
// - OOB error in current tree
// - OOB error mean in current tree
// - TP
// - FP
// - TN
// - FN
// - TP rate
// - FP rate
// - TN rate
// - Precision
// - F-measure
// - Accuracy
//
// and total of all of them.
//
func (forest *Runtime) writeStat(stat *classifiers.Stat) error {
	if forest.statWriter == nil {
		return nil
	}
	if stat == nil {
		return nil
	}
	return forest.statWriter.WriteRawRow(stat.ToRow(), nil, nil)
}

//
// closeStatsFile will close statistics file for writing.
//
func (forest *Runtime) closeStatsFile() (e error) {
	if forest.statWriter == nil {
		return
	}

	e = forest.statWriter.Close()
	forest.statWriter = nil

	return
}
