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
	"github.com/golang/glog"
	"github.com/shuLhan/go-mining/classifiers/cart"
	"github.com/shuLhan/go-mining/dataset"
)

const (
	// DefNumTree default number of tree.
	DefNumTree = 100
	// DefNumSubsample default percentage of sample that will be used for
	// bootstraping a tree.
	DefNumSubsample = 66
)

var (
	// ErrNoInput will tell you when no input is given.
	ErrNoInput = errors.New("randomforest: input reader is empty")
)

/*
Ensemble contains input and output configuration when generating random forest.
*/
type Ensemble struct {
	// Trees contain all tree in the forest.
	Trees []cart.Input
	// NSubsample number of subsample for bootstraping.
	NSubsample int
	// NFeature number of feature randomly selected for each tree.
	NFeature int
	// NTree number of tree in forest.
	NTree int
	// OOBErrMeanVal contain the average out-of-back error value.
	OOBErrMeanVal float64
}

/*
init check and initialize forest input and attributes.
*/
func (forest *Ensemble) init(samples *dataset.Reader, ntree int, nfeature int,
	npercent int) error {
	// check input samples
	if samples == nil {
		return ErrNoInput
	}

	// Check number of feature, tree, and bootstrap percentage.
	// If its not set, set to default value.
	if nfeature <= 0 {
		nfeature = samples.GetNColumn() - 1
	}
	if ntree <= 0 {
		ntree = DefNumTree
	}
	if npercent <= 0 {
		npercent = DefNumSubsample
	}

	forest.NFeature = nfeature
	forest.NTree = ntree

	// count number of samples
	forest.NSubsample = int(float32(samples.GetNRow()) *
		(float32(npercent) / 100.0))

	forest.Trees = make([]cart.Input, 0)

	return nil
}

/*
AddCartTree add tree to forest
*/
func (forest *Ensemble) AddCartTree(tree cart.Input) {
	forest.Trees = append(forest.Trees, tree)
}

/*
Ensembling the forest using samples dataset.

- samples: as original dataset.
- ntree: number of tree to be generated.
- nfeature: number of feature randomly selected for each tree for splitting
(the `m`).
- npercent: percentage of subset that will be taken randomly for
generating a tree.
*/
func Ensembling(samples *dataset.Reader, ntree int, nfeature int,
	npercent int) (
	forest Ensemble,
	ooberrsteps []float64,
	e error,
) {
	e = forest.init(samples, ntree, nfeature, npercent)
	if e != nil {
		return
	}

	glog.V(1).Info(">>> forest:", forest)

	ooberrsteps = make([]float64, ntree)
	totalOOBErr := 0.0

	// create trees
	for t := 0; t < ntree; t++ {
		// select random samples with replacement.
		pickedRows, unpickedRows, _, _, e := samples.RandomPickRows(
			forest.NSubsample, true)

		if e != nil {
			return forest, ooberrsteps, e
		}

		glog.V(2).Info(">>> picked rows:", pickedRows)

		// select random features, without replacement, not including
		// target feature.
		pickedCols, _, pickedColsIdx, _, e := pickedRows.RandomPickColumns(
			forest.NFeature, false)

		if e != nil {
			return forest, ooberrsteps, e
		}

		glog.V(2).Info(">>> picked cols:", pickedCols)

		// build tree.
		cart := cart.Input{
			SplitMethod: cart.SplitMethodGini,
		}

		e = cart.BuildTree(&pickedCols)
		if e != nil {
			return forest, ooberrsteps, e
		}

		glog.V(2).Info(">>> TREE:", &cart)

		glog.V(2).Info(">>> unpicked rows:", unpickedRows)
		glog.V(2).Info(">>> picked cols idx:", pickedColsIdx)

		// run OOB on tree.
		oob, e := unpickedRows.SelectColumnsByIdx(pickedColsIdx)
		if e != nil {
			return forest, ooberrsteps, e
		}

		glog.V(2).Info(">>> OOB :", oob)

		ooberr, e := cart.CountOOBError(oob)

		if e != nil {
			return forest, ooberrsteps, e
		}

		glog.V(2).Infof(">>> t %d - OOB error: %f", t, ooberr)

		// calculate error.
		totalOOBErr += ooberr

		ooberrsteps[t] = totalOOBErr / float64(t+1)

		// Add tree to forest.
		forest.AddCartTree(cart)
	}

	forest.OOBErrMeanVal = totalOOBErr / float64(ntree)

	glog.V(1).Info(">>> OOB error mean: ", forest.OOBErrMeanVal)

	return forest, ooberrsteps, nil
}
