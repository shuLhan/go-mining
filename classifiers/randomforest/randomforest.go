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
	"github.com/shuLhan/dsv/util"
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
	// bagIndices contain list of selected samples at bootstraping.
	bagIndices [][]int
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
AddBagIndex add bagging index for book keeping.
*/
func (forest *Ensemble) AddBagIndex(bagIndex []int) {
	forest.bagIndices = append(forest.bagIndices, bagIndex)
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
		bootstrap, oob, bagIdx, oobIdx, e := samples.RandomPickRows(
			forest.NSubsample, true)

		if e != nil {
			return forest, ooberrsteps, e
		}

		glog.V(3).Info(">>> picked rows:", bootstrap)
		glog.V(3).Info(">>> unpicked rows:", oob)

		// build tree.
		cart := cart.Input{
			SplitMethod:    cart.SplitMethodGini,
			NRandomFeature: nfeature,
		}

		e = cart.BuildTree(&bootstrap)
		if e != nil {
			return forest, ooberrsteps, e
		}

		glog.V(3).Info(">>> TREE:", &cart)

		// Add tree to forest.
		forest.AddCartTree(cart)
		forest.AddBagIndex(bagIdx)

		// run OOB on current tree.
		ooberr := forest.ClassifySet(&oob, oobIdx)

		// calculate error.
		totalOOBErr += ooberr
		ooberrsteps[t] = totalOOBErr / float64(t+1)

		glog.V(2).Info(">>> tree #", t, " - OOB error: ", ooberr,
			" total OOB error: ", ooberrsteps[t])

	}

	forest.OOBErrMeanVal = totalOOBErr / float64(ntree)

	glog.V(1).Info(">>> OOB error mean: ", forest.OOBErrMeanVal)

	return forest, ooberrsteps, nil
}

/*
ClassifySet given a dataset predict their class by running each sample in
forest. Return miss classification rate:

	(number of missed class / number of samples).
*/
func (forest *Ensemble) ClassifySet(dataset *dataset.Reader, dsIdx []int) (
	missrate float64,
) {
	var class string
	targetClass := dataset.GetTargetClass()
	targetAttr := dataset.GetTarget()
	origTarget := targetAttr.ToStringSlice()
	indexlen := len(dsIdx)

	targetAttr.ClearValues()

	for x, row := range dataset.Rows {
		var votes []string

		for y, tree := range forest.Trees {
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

		class = util.StringsGetMajority(votes, targetClass)
		(*targetAttr).Records[x].V = class
	}

	glog.V(2).Info(">>> target    : ", origTarget)
	glog.V(2).Info(">>> prediction: ", targetAttr.ToStringSlice())

	missrate = util.CountMissRate(origTarget, targetAttr.ToStringSlice())

	// set original target values back.
	targetAttr.SetValues(origTarget)

	return missrate
}
