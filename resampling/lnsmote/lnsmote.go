// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package lnsmote implement the Local-Neighborhood algorithm from the paper,

	Maciejewski, Tomasz, and Jerzy Stefanowski. "Local neighbourhood
	extension of SMOTE for mining imbalanced data." Computational
	Intelligence and Data Mining (CIDM), 2011 IEEE Symposium on. IEEE,
	2011.
*/
package lnsmote

import (
	"github.com/golang/glog"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/knn"
	"math/rand"
	"time"
)

/*
LNSmote parameters for input and output.
*/
type Input struct {
	// Input the K-Nearest-Neighbourhood parameters.
	knn.Input
	// ClassMinor the minority sample in dataset
	ClassMinor string
	// PercentOver input for oversampling percentage.
	PercentOver int
	// n input for number of new synthetic per sample.
	n int
	// Synthetic output for new sample.
	Synthetic dsv.Dataset
	// minority contain minor class in samples.
	minority dsv.Dataset
	// dataset contain all samples
	dataset dsv.Dataset
}

func (in *Input) Init(dataset dsv.Dataset) {
	// Count number of sythetic sample that will be created.
	if in.PercentOver < 100 {
		in.PercentOver = 100
	}

	in.n = in.PercentOver / 100.0
	in.dataset = dataset

	in.minority = dataset.SelectRowsWhere(in.ClassIdx, in.ClassMinor)

	glog.V(1).Info(">>> n: ", in.n)
	glog.V(1).Info(">>> n minority: ", in.minority.Len())
}

func (in *Input) Resampling(dataset dsv.Dataset) (synthetics dsv.Dataset) {
	in.Init(dataset)

	for x, p := range in.minority.Rows {
		neighbors := in.FindNeighbors(in.dataset.Rows, p)

		glog.V(3).Info(">>> neighbors:", neighbors.Rows)

		for y := 0; y < in.n; y++ {
			syn := in.createSynthetic(p, neighbors)

			// no synthetic can be created, increase neighbours
			// range.
			if syn != nil {
				in.Synthetic.PushRow(syn)
			}
		}

		glog.Infof(">>> %-4d n synthetics: %v", x, in.Synthetic.Len())

		if glog.V(2) {
			time.Sleep(5000 * time.Millisecond)
		}
	}

	return in.Synthetic
}

func (in *Input) createSynthetic(p dsv.Row, neighbors knn.Neighbors) (
	synthetic dsv.Row,
) {
	rand.Seed(time.Now().UnixNano())

	// choose one of the K nearest neighbors
	randIdx := rand.Intn(neighbors.Len())
	n := neighbors.GetRow(randIdx)

	// Check if synthetic sample can be created from p and n.
	canit, slp, sln := in.canCreate(p, *n)
	if !canit {
		glog.V(2).Info(">>> can not create synthetic")
		// we can not create from p and synthetic.
		return nil
	}

	synthetic = p.Clone()

	for x, srec := range synthetic {
		// Skip class attribute.
		if x == in.ClassIdx {
			continue
		}

		delta := in.randomGap(p, *n, slp.Len(), sln.Len())
		pv := p[x].Value().(float64)
		diff := (*n)[x].Value().(float64) - pv
		srec.SetFloat(pv + delta*diff)
	}

	return
}

func (in *Input) canCreate(p, n dsv.Row) (bool, dsv.Dataset, dsv.Dataset) {
	slp := in.safeLevel(p)
	sln := in.safeLevel2(p, n)

	glog.V(2).Info(">>> slp : ", slp.Len())
	glog.V(2).Info(">>> sln : ", sln.Len())

	return slp.Len() != 0 || sln.Len() != 0, slp, sln
}

func (in *Input) safeLevel(p dsv.Row) dsv.Dataset {
	neighbors := in.FindNeighbors(in.dataset.Rows, p)
	minorNeighbors := neighbors.SelectRowsWhere(in.ClassIdx, in.ClassMinor)

	return minorNeighbors
}

func (in *Input) safeLevel2(p, n dsv.Row) dsv.Dataset {
	neighbors := in.FindNeighbors(in.dataset.Rows, n)

	// check if n is in minority class.
	nIsMinor := n[in.ClassIdx].IsEqual(in.ClassMinor)

	// check if p is in neighbors.
	pInNeighbors, pidx := neighbors.Rows.Contain(p)

	// if p in neighbors, replace it with neighbours in K+1
	if nIsMinor && pInNeighbors {
		glog.V(1).Info(">>> Replacing ", pidx)
		glog.V(2).Info(">>> Replacing ", pidx, " in ", neighbors)

		repl := in.AllNeighbors.GetRow(in.K + 1)
		neighbors.Rows[pidx] = *repl

		glog.V(2).Info(">>> Replacement ", neighbors)
	}

	minorNeighbors := neighbors.SelectRowsWhere(in.ClassIdx, in.ClassMinor)

	return minorNeighbors
}

func (in *Input) randomGap(p, n dsv.Row, lenslp, lensln int) (delta float64) {
	if lensln == 0 && lenslp > 0 {
		return
	}

	slratio := float64(lenslp) / float64(lensln)
	if slratio == 1 {
		delta = rand.Float64()
	} else if slratio > 1 {
		delta = rand.Float64() * (1 / slratio)
	} else {
		delta = 1 - rand.Float64()*slratio
	}

	return delta
}
