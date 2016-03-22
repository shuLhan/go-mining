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
	"github.com/shuLhan/go-mining/classifiers"
	"github.com/shuLhan/go-mining/classifiers/cart"
	"github.com/shuLhan/tabula"
	"github.com/shuLhan/tabula/util"
	"github.com/shuLhan/tekstus"
	"os"
	"strconv"
)

const (
	// DefNumTree default number of tree.
	DefNumTree = 100
	// DefPercentBoot default percentage of sample that will be used for
	// bootstraping a tree.
	DefPercentBoot = 66
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
	// nSubsample number of samples used for bootstraping.
	nSubsample int
	// oobErrorTotal contain the total of out-of-bag error value.
	oobErrorTotal float64
	// oobErrorSteps contain OOB error for each steps building a forest
	// from the first tree until `n` tree.
	oobErrorSteps []float64
	// oobErrorStepsMean contain mean for OOB error for each steps, using
	// oobErrorTotal / current-number-of-tree
	oobErrorStepsMean []float64
	// oobErrorStats contain OOB statistics from the first tree until NTree.
	oobErrorStats []classifiers.ConfusionMatrix
	// trees contain all tree in the forest.
	trees []cart.Runtime
	// bagIndices contain list of index of selected samples at bootstraping
	// for book-keeping.
	bagIndices [][]int
	// cm contain class error.
	cm classifiers.ConfusionMatrix
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
AddOobErrorStats will append new result of OOB statistics.
*/
func (forest *Runtime) AddOobErrorStats(stats *classifiers.ConfusionMatrix) {
	forest.oobErrorStats = append(forest.oobErrorStats, *stats)
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

/*
Init will check forest inputs and set it to default values if invalid.
*/
func (forest *Runtime) Init() {
	if forest.NTree <= 0 {
		forest.NTree = DefNumTree
	}
	if forest.PercentBoot <= 0 {
		forest.PercentBoot = DefPercentBoot
	}
}

/*
Build the forest using samples dataset.

Algorithm,

(0) Recheck input value: number of tree, percentage bootstrap.
(1) Calculate number of random samples for each tree.

	number-of-sample * percentage-of-bootstrap

(2) Grow trees
(2.1) Create new tree
(2.2) Save current tree OOB error value.
(2.3) Recalculate total OOB error and total error mean.
*/
func (forest *Runtime) Build(samples tabula.ClasetInterface) (e error) {
	// check input samples
	if samples == nil {
		return ErrNoInput
	}

	// (0)
	forest.Init()

	// (1)
	forest.nSubsample = int(float32(samples.GetNRow()) *
		(float32(forest.PercentBoot) / 100.0))

	if DEBUG >= 1 {
		fmt.Println("[randomforest] forest:", forest)
	}

	oobErr := 0.0

	// (2)
	for t := 0; t < forest.NTree; t++ {
		// (2.1)
		oobErr, e = forest.GrowTree(samples)

		// (2.2)
		forest.oobErrorSteps = append(forest.oobErrorSteps, oobErr)

		// (2.3)
		forest.oobErrorTotal += oobErr

		oobErrStepMean := forest.oobErrorTotal / float64(t+1)
		forest.oobErrorStepsMean = append(forest.oobErrorStepsMean,
			oobErrStepMean)

		if DEBUG >= 2 {
			fmt.Printf("[randomforest] tree #%4d"+
				" OOB error: %.4f,total OOB error: %.4f\n",
				t, oobErr, forest.oobErrorSteps[t])
		}
	}

	if DEBUG >= 1 {
		fmt.Println("[randomforest] OOB error mean: ",
			forest.OobErrorTotalMean())
	}

	return nil
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
(6) Calculate OOB error rate.
*/
func (forest *Runtime) GrowTree(samples tabula.ClasetInterface) (
	oobErr float64,
	e error,
) {
	// (1)
	bag, oob, bagIdx, oobIdx := tabula.RandomPickRows(
		samples.(tabula.DatasetInterface),
		forest.nSubsample, true)

	// (2)
	bagset := bag.(tabula.ClasetInterface)
	cart, e := cart.New(bagset, cart.SplitMethodGini, forest.NRandomFeature)
	if e != nil {
		return
	}

	// (3)
	forest.AddCartTree(*cart)

	// (4)
	forest.AddBagIndex(bagIdx)

	// (5)
	oobset := oob.(tabula.ClasetInterface)
	cm := forest.ClassifySet(oobset, oobIdx, false)
	forest.AddOobErrorStats(cm)

	// (6)
	oobErr = cm.GetFalseRate()

	return
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
(3.1.1) classify row in tree,
(3.1.2) save tree class value.
(3.2) Collect majority class vote in forest.
(4) Compute confusion matrix from predictions.
(5) Restore original target values in testset.
*/
func (forest *Runtime) ClassifySet(testset tabula.ClasetInterface,
	testsetIdx []int, uniq bool) (
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
	rowsused := 0
	for x, row := range *rows {
		var votes []string

		// (3.1)
		for y, tree := range forest.trees {
			if uniq && indexlen > 0 {
				// check if sample index is used to build the
				// tree
				exist := util.IntIsExist(forest.bagIndices[y],
					testsetIdx[x])
				if exist {
					continue
				}
			}
			if y == 0 {
				rowsused++
			}

			// (3.1.1)
			class := tree.Classify(row)

			// (3.1.2)
			votes = append(votes, class)
		}

		// (3.2)
		class := tekstus.WordsMaxCountOf(votes, targetVS, false)
		(*targetAttr).Records[x].V = class
	}

	if DEBUG >= 2 {
		fmt.Printf("[randomforest] oob samples: %d, oob used: %d\n",
			len(*rows), rowsused)
	}

	// (4)
	predictions := targetAttr.ToStringSlice()

	cm = classifiers.NewConfusionMatrix(targetVS, targetValues,
		predictions)

	// (5)
	targetAttr.SetValues(targetValues)

	return cm
}
