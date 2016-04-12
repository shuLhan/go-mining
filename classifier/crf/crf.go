// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package crf implement the cascaded random forest algorithm, proposed by
Baumann et.al in their paper:

	Baumann, Florian, et al. "Cascaded Random Forest for Fast Object
	Detection." Image Analysis. Springer Berlin Heidelberg, 2013. 131-142.

*/
package crf

import (
	"errors"
	"fmt"
	"github.com/shuLhan/go-mining/classifier"
	"github.com/shuLhan/go-mining/classifier/rf"
	"github.com/shuLhan/tabula"
	"math"
	"os"
	"strconv"
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
	DefStatsFile = "crf.stats"
)

var (
	// DEBUG level, can set from environment CRF_DEBUG variable.
	DEBUG = 0
)

var (
	// ErrNoInput will tell you when no input is given.
	ErrNoInput = errors.New("rf: input samples is empty")
)

/*
Runtime define the cascaded random forest runtime input and output.
*/
type Runtime struct {
	// Runtime embed common fields for classifier.
	classifier.Runtime

	// NStage number of stage.
	NStage int `json:"NStage"`
	// TPRate threshold for true positive rate per stage.
	TPRate float64 `json:"TPRate"`
	// TNRate threshold for true negative rate per stage.
	TNRate float64 `json:"TNRate"`

	// NTree number of tree in each stage.
	NTree int `json:"NTree"`
	// NRandomFeature number of features used to split the dataset.
	NRandomFeature int `json:"NRandomFeature"`
	// PercentBoot percentage of bootstrap.
	PercentBoot int `json:"PercentBoot"`

	// forests contain forest for each stage.
	forests []*rf.Runtime
	// weights contain weight for each stage.
	weights []float64
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

	return crf
}

//
// AddForest will append new forest.
//
func (crf *Runtime) AddForest(forest *rf.Runtime) {
	crf.forests = append(crf.forests, forest)
}

//
// Initialize will check crf inputs and set it to default values if its
// invalid.
//
func (crf *Runtime) Initialize(samples tabula.ClasetInterface) error {
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

	return crf.Runtime.Initialize()
}

//
// Build given a sample dataset, build the stage with randomforest.
//
func (crf *Runtime) Build(samples tabula.ClasetInterface) (e error) {
	if samples == nil {
		return ErrNoInput
	}

	e = crf.Initialize(samples)
	if e != nil {
		return
	}

	if DEBUG >= 1 {
		fmt.Println("[crf] # samples:", samples.Len())
		fmt.Println("[crf] sample:", samples.GetRow(0))
		fmt.Println("[crf]", crf)
	}

	for x := 0; x < crf.NStage; x++ {
		if DEBUG >= 1 {
			fmt.Printf("====\n[crf] stage # %d\n", x)
		}

		forest, e := crf.createForest(samples)
		if e != nil {
			return e
		}

		e = crf.finalizeStage(forest)
		if e != nil {
			return e
		}
	}

	return crf.Finalize()
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
// (3) Calculate weight.
// (4) Delete true-negative from samples.
// (5) Refill samples with false-positive.
//
func (crf *Runtime) createForest(samples tabula.ClasetInterface) (
	forest *rf.Runtime, e error,
) {
	var cm *classifier.CM
	var stat *classifier.Stat

	// (1)
	forest = rf.New(crf.NTree, crf.NRandomFeature, crf.PercentBoot)

	e = forest.Initialize(samples)
	if e != nil {
		return nil, e
	}

	// (2)
	for t := 0; t < crf.NTree; t++ {
		if DEBUG >= 1 {
			fmt.Printf("[crf] tree # %d\n", t)
		}

		// (2.1)
		for {
			cm, stat, e = forest.GrowTree(samples)
			if e == nil {
				break
			}
		}

		// (2.2)
		if stat.TPRate > crf.TPRate &&
			stat.TNRate > crf.TNRate {
			break
		}
	}

	e = forest.Finalize()
	if e != nil {
		return nil, e
	}

	// (3)
	crf.computeWeight(stat)

	// (4)
	crf.deleteTrueNegative(samples, cm)

	// (5)
	// crf.refillWithFalsePositive(samples, cm)

	return forest, nil
}

//
// finalizeStage save forest and write the forest statistic to file.
//
func (crf *Runtime) finalizeStage(forest *rf.Runtime) (e error) {
	stat := forest.StatTotal()
	stat.ID = int64(len(crf.forests))

	e = crf.WriteStat(stat)
	if e != nil {
		return e
	}

	crf.AddStat(stat)
	crf.ComputeStatTotal(stat)

	if DEBUG >= 1 {
		crf.PrintStatTotal(nil)
	}

	// (7)
	crf.AddForest(forest)

	return nil
}

//
// computeWeight will compute the weight of stage based on F-measure of the
// last tree in forest.
//
func (crf *Runtime) computeWeight(stat *classifier.Stat) {
	crf.weights = append(crf.weights, math.Exp(stat.FMeasure))
}

func (crf *Runtime) deleteTrueNegative(samples tabula.ClasetInterface,
	cm *classifier.CM,
) {

}
