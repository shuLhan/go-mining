// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
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
	NTree int
	// NFeature number of feature randomly selected for each tree.
	NFeature int
	// PercentBoot percentage of sample for bootstraping.
	PercentBoot int
	// NSubsample number of samples used for bootstraping.
	NSubsample int
	// OobErrMeanVal contain the average out-of-bag error value.
	OobErrMeanVal float64
	// OobErrSteps contain OOB error for each steps building a forest from
	// the first tree until n tree.
	OobErrSteps []float64
	// OobStats contain OOB statistics from the first tree until NTree.
	OobStats []classifiers.TestStats
	// trees contain all tree in the forest.
	trees []cart.Runtime
	// bagIndices contain list of index of selected samples at bootstraping
	// for book-keeping.
	bagIndices [][]int
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
		NTree:       ntree,
		NFeature:    nfeature,
		PercentBoot: percentboot,
		OobErrSteps: make([]float64, ntree),
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
AddOOBStats will append new result of OOB statistics.
*/
func (forest *Runtime) AddOobStats(stats classifiers.TestStats) {
	forest.OobStats = append(forest.OobStats, stats)
}

/*
Build the forest using samples dataset.

- samples: as original dataset.
*/
func (forest *Runtime) Build(samples tabula.ClasetInterface) (e error) {
	// check input samples
	if samples == nil {
		return ErrNoInput
	}

	// Calculate number of subsample that will be used randomly in each
	// tree.
	forest.NSubsample = int(float32(samples.GetNRow()) *
		(float32(forest.PercentBoot) / 100.0))

	if DEBUG >= 1 {
		fmt.Println("[randomforest] forest:", forest)
	}

	totalOobErr := 0.0
	oobErr := 0.0

	// create trees
	for t := 0; t < forest.NTree; t++ {
		oobErr, e = forest.GrowTree(samples)
		totalOobErr += oobErr
		forest.OobErrSteps[t] = totalOobErr / float64(t+1)

		if DEBUG >= 1 {
			fmt.Printf("[randomforest] tree #%4d"+
				" OOB error: %.4f,"+
				" total OOB error: %.4f\n",
				t, oobErr,
				forest.OobErrSteps[t])
		}
	}

	forest.OobErrMeanVal = totalOobErr / float64(forest.NTree)

	if DEBUG >= 1 {
		fmt.Println("[randomforest] OOB error mean: ",
			forest.OobErrMeanVal)
	}

	return nil
}

func (forest *Runtime) GrowTree(samples tabula.ClasetInterface) (
	oobErr float64,
	e error,
) {
	// Select random samples with replacement.
	bag, oob, bagIdx, oobIdx := tabula.RandomPickRows(
		samples.(tabula.DatasetInterface),
		forest.NSubsample, true)

	// Build tree using CART.
	ds := bag.(tabula.ClasetInterface)
	cart, e := cart.New(ds, cart.SplitMethodGini, forest.NFeature)
	if e != nil {
		return
	}

	// Add tree to forest.
	forest.AddCartTree(*cart)
	forest.AddBagIndex(bagIdx)

	// Run OOB on current forest.
	oobset := oob.(tabula.ClasetInterface)
	testStats := forest.ClassifySet(oobset, oobIdx)

	// Calculate OOB error rate.
	oobErr = testStats.GetFPRate()

	forest.AddOobStats(testStats)

	return
}

/*
ClassifySet given a dataset predict their class by running each sample in
forest. Return miss classification rate:

	(number of missed class / number of samples).
*/
func (forest *Runtime) ClassifySet(dataset tabula.ClasetInterface, dsIdx []int) (
	testStats classifiers.TestStats,
) {
	var class string
	targetClass := dataset.GetClassValueSpace()
	targetAttr := dataset.GetClassColumn()
	origTarget := targetAttr.ToStringSlice()
	indexlen := len(dsIdx)

	targetAttr.ClearValues()

	rows := dataset.GetRows()
	for x, row := range *rows {
		var votes []string

		for y, tree := range forest.trees {
			if indexlen > 0 {
				// check if sample index is used to build the
				// tree
				exist := util.IntIsExist(forest.bagIndices[y],
					dsIdx[x])
				if exist {
					continue
				}
			}

			class = tree.Classify(row)
			votes = append(votes, class)
		}

		class = tekstus.WordsMaxCountOf(votes, targetClass, false)
		(*targetAttr).Records[x].V = class
	}

	predictions := targetAttr.ToStringSlice()

	if DEBUG >= 2 {
		fmt.Println("[randomforest] target    : ", origTarget)
		fmt.Println("[randomforest] prediction: ", predictions)
	}

	_, testStats.NNegative, testStats.NSample = tekstus.WordsCountMissRate(
		origTarget, predictions)

	testStats.NPositive = testStats.NSample - testStats.NNegative

	// Set original target values back.
	targetAttr.SetValues(origTarget)

	return testStats
}
