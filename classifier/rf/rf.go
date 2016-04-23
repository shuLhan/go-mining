// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package rf implement ensemble of classifiers using random forest
algorithm by Breiman and Cutler.

	Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.

The implementation is based on various sources and using author experience.
*/
package rf

import (
	"errors"
	"fmt"
	"github.com/shuLhan/go-mining/classifier"
	"github.com/shuLhan/go-mining/classifier/cart"
	"github.com/shuLhan/numerus"
	"github.com/shuLhan/tabula"
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
	DefStatsFile = "rf.stats"
)

var (
	// DEBUG level, set it from environment variable.
	DEBUG = 0
)

var (
	// ErrNoInput will tell you when no input is given.
	ErrNoInput = errors.New("rf: input samples is empty")
)

/*
Runtime contains input and output configuration when generating random forest.
*/
type Runtime struct {
	// Runtime embed common fields for classifier.
	classifier.Runtime

	// NTree number of tree in forest.
	NTree int `json:"NTree"`
	// NRandomFeature number of feature randomly selected for each tree.
	NRandomFeature int `json:"NRandomFeature"`
	// PercentBoot percentage of sample for bootstraping.
	PercentBoot int `json:"PercentBoot"`
	// RunOOB if its true the OOB will be computed.
	RunOOB bool `json:"RunOOB"`

	// nSubsample number of samples used for bootstraping.
	nSubsample int
	// trees contain all tree in the forest.
	trees []cart.Runtime
	// bagIndices contain list of index of selected samples at bootstraping
	// for book-keeping.
	bagIndices [][]int
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
func New(ntree, nfeature, percentboot int, runOOB bool) (
	forest *Runtime,
) {
	forest = &Runtime{
		NTree:          ntree,
		NRandomFeature: nfeature,
		PercentBoot:    percentboot,
		RunOOB:         runOOB,
	}

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

//
// Initialize will check forest inputs and set it to default values if invalid.
//
// It will also calculate number of random samples for each tree using,
//
//	number-of-sample * percentage-of-bootstrap
//
//
func (forest *Runtime) Initialize(samples tabula.ClasetInterface) error {
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

	forest.nSubsample = int(float32(samples.GetNRow()) *
		(float32(forest.PercentBoot) / 100.0))

	return forest.Runtime.Initialize()
}

/*
Build the forest using samples dataset.

Algorithm,

(0) Recheck input value: number of tree, percentage bootstrap, etc; and
    Open statistic file output.
(1) For 0 to NTree,
(1.1) Create new tree, repeat until all trees has been build.
(2) Compute and write total statistic.
*/
func (forest *Runtime) Build(samples tabula.ClasetInterface) (e error) {
	// check input samples
	if samples == nil {
		return ErrNoInput
	}

	// (0)
	e = forest.Initialize(samples)
	if e != nil {
		return
	}

	if DEBUG >= 1 {
		fmt.Println("[rf] forest:", forest)
	}

	// (1)
	for t := 0; t < forest.NTree; t++ {
		if DEBUG >= 1 {
			fmt.Printf("----\n[rf] tree # %d\n", t)
		}

		// (1.1)
		for {
			_, _, e = forest.GrowTree(samples)
			if e == nil {
				break
			}

			fmt.Println("[rf] error: ", e)
		}
	}

	// (2)
	return forest.Finalize()
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
	cm *classifier.CM, stat *classifier.Stat, e error,
) {
	stat = &classifier.Stat{}
	stat.ID = int64(len(forest.trees))
	stat.StartTime = time.Now().Unix()

	// (1)
	bag, oob, bagIdx, oobIdx := tabula.RandomPickRows(
		samples.(tabula.DatasetInterface),
		forest.nSubsample, true)

	bagset := bag.(tabula.ClasetInterface)

	if DEBUG >= 2 {
		bagset.RecountMajorMinor()
		fmt.Println("[rf] Bagging:", bagset)
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
	if forest.RunOOB {
		oobset := oob.(tabula.ClasetInterface)
		_, cm = forest.ClassifySet(oobset, oobIdx, true)

		forest.AddCM(cm)
	}

	stat.EndTime = time.Now().Unix()
	stat.ElapsedTime = stat.EndTime - stat.StartTime

	if DEBUG >= 1 {
		fmt.Printf("[rf] Elapsed time: %d s\n",
			stat.ElapsedTime)
	}

	forest.AddStat(stat)

	// (6)
	if forest.RunOOB {
		forest.ComputeStatFromCM(stat, cm)
		forest.ComputeStatTotal(stat)

		e = forest.WriteStat(stat)
	}

	return cm, stat, e
}

/*
ClassifySet given a dataset predict their class by running each sample in
forest. Return miss classification rate:

	(number of missed class / number of samples).

Algorithm,

(0) Get value space (possible class values in dataset)
(1) For each row in test-set,
(1.1) collect votes in all trees, and
(1.2) select majority class vote.
(2) Compute confusion matrix from predictions.
*/
func (forest *Runtime) ClassifySet(testset tabula.ClasetInterface,
	testsetIds []int, uniq bool,
) (
	predicts []string, cm *classifier.CM,
) {
	// (0)
	vs := testset.GetClassValueSpace()

	// (1)
	rows := testset.GetRows()
	for x, row := range *rows {
		// (1.1)
		votes := forest.Votes(row, testsetIds[x], uniq)

		// (1.2)
		majorVote := tekstus.WordsMaxCountOf(votes, vs, false)

		predicts = append(predicts, majorVote)
	}

	// (2)
	actuals := testset.GetClassAsStrings()
	cm = forest.ComputeCM(testsetIds, vs, actuals, predicts)

	return predicts, cm
}

//
// Votes will return votes, or classes, in each tree based on sample.
// If uniq is true then the `sampleIdx` will be checked in if it has been used
// when training the tree, if its exist then the sample will be skipped.
//
// (1) If row is used to build the tree then skip it,
// (2) classify row in tree,
// (3) save tree class value.
//
func (forest *Runtime) Votes(sample *tabula.Row, sampleIdx int, uniq bool) (
	votes []string,
) {
	for x, tree := range forest.trees {
		// (1)
		if uniq && sampleIdx > 0 {
			exist := numerus.IntsIsExist(forest.bagIndices[x],
				sampleIdx)
			if exist {
				continue
			}
		}

		// (2)
		class := tree.Classify(sample)

		// (3)
		votes = append(votes, class)
	}
	return votes
}
