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
	"fmt"
	"github.com/shuLhan/go-mining/knn"
	"github.com/shuLhan/tabula"
	"math/rand"
	"os"
	"strconv"
	"time"
)

var (
	// DEBUG debug level, set from environment.
	DEBUG = 0
)

//
// Runtime parameters for input and output.
//
type Runtime struct {
	// Input the K-Nearest-Neighbourhood parameters.
	knn.Input
	// ClassMinor the minority sample in dataset
	ClassMinor string
	// PercentOver input for oversampling percentage.
	PercentOver int
	// n input for number of new synthetic per sample.
	n int
	// Synthetic output for new sample.
	Synthetic tabula.Dataset
	// minority contain minor class in samples.
	minority tabula.Dataset
	// dataset contain all samples
	dataset tabula.Dataset
}

func init() {
	v := os.Getenv("LNSMOTE_DEBUG")
	if v == "" {
		DEBUG = 0
	} else {
		DEBUG, _ = strconv.Atoi(v)
	}
}

//
// Init will initialize LNSmote object by checking input values and set it to
// default if not set or invalid.
//
func (in *Runtime) Init(dataset tabula.DatasetInterface) {
	// Count number of sythetic sample that will be created.
	if in.PercentOver < 100 {
		in.PercentOver = 100
	}

	in.n = in.PercentOver / 100.0
	in.dataset = *(dataset.(*tabula.Dataset))

	in.minority = *tabula.SelectRowsWhere(dataset, in.ClassIdx,
		in.ClassMinor).(*tabula.Dataset)

	if DEBUG >= 1 {
		fmt.Println("[lnsmote] n:", in.n)
		fmt.Println("[lnsmote] n minority:", in.minority.Len())
	}
}

//
// Resampling will run resampling process on dataset and return the synthetic
// samples.
//
func (in *Runtime) Resampling(dataset tabula.DatasetInterface) (
	synthetics tabula.DatasetInterface,
) {
	in.Init(dataset)

	for x, p := range in.minority.Rows {
		neighbors := in.FindNeighbors(in.dataset.Rows, p)

		if DEBUG >= 3 {
			fmt.Println("[lnsmote] neighbors:", neighbors.Rows)
		}

		for y := 0; y < in.n; y++ {
			syn := in.createSynthetic(p, neighbors)

			// no synthetic can be created, increase neighbours
			// range.
			if syn != nil {
				in.Synthetic.PushRow(syn)
			}
		}

		if DEBUG >= 1 {
			fmt.Printf("[lnsmote] %-4d n synthetics: %v", x,
				in.Synthetic.Len())
		}

		if DEBUG >= 2 {
			time.Sleep(5000 * time.Millisecond)
		}
	}

	return &in.Synthetic
}

//
// createSynthetic will create synthetics row from original row `p` and their
// `neighbors`.
//
func (in *Runtime) createSynthetic(p tabula.Row, neighbors knn.Neighbors) (
	synthetic tabula.Row,
) {
	rand.Seed(time.Now().UnixNano())

	// choose one of the K nearest neighbors
	randIdx := rand.Intn(neighbors.Len())
	n := neighbors.GetRow(randIdx)

	// Check if synthetic sample can be created from p and n.
	canit, slp, sln := in.canCreate(p, *n)
	if !canit {
		if DEBUG >= 2 {
			fmt.Println("[lnsmote] can not create synthetic")
		}
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

//
// canCreate return true if synthetic can be created between two sample `p` and
// `n`. Otherwise it will return false.
//
func (in *Runtime) canCreate(p, n tabula.Row) (bool, tabula.DatasetInterface,
	tabula.DatasetInterface,
) {
	slp := in.safeLevel(p)
	sln := in.safeLevel2(p, n)

	if DEBUG >= 2 {
		fmt.Println("[lnsmote] slp : ", slp.Len())
		fmt.Println("[lnsmote] sln : ", sln.Len())
	}

	return slp.Len() != 0 || sln.Len() != 0, slp, sln
}

//
// safeLevel return the minority neighbors in sample `p`.
//
func (in *Runtime) safeLevel(p tabula.Row) tabula.DatasetInterface {
	neighbors := in.FindNeighbors(in.dataset.Rows, p)
	minorNeighbors := tabula.SelectRowsWhere(&neighbors, in.ClassIdx,
		in.ClassMinor)

	return minorNeighbors
}

//
// safeLevel2 return the minority neighbors between sample `p` and `n`.
//
func (in *Runtime) safeLevel2(p, n tabula.Row) tabula.DatasetInterface {
	neighbors := in.FindNeighbors(in.dataset.Rows, n)

	// check if n is in minority class.
	nIsMinor := n[in.ClassIdx].IsEqual(in.ClassMinor)

	// check if p is in neighbors.
	pInNeighbors, pidx := neighbors.Rows.Contain(p)

	// if p in neighbors, replace it with neighbours in K+1
	if nIsMinor && pInNeighbors {
		if DEBUG >= 1 {
			fmt.Println("[lnsmote] Replacing ", pidx)
		}
		if DEBUG >= 2 {
			fmt.Println("[lnsmote] Replacing ", pidx, " in ", neighbors)
		}

		repl := in.AllNeighbors.GetRow(in.K + 1)
		neighbors.Rows[pidx] = *repl

		if DEBUG >= 2 {
			fmt.Println("[lnsmote] Replacement ", neighbors)
		}
	}

	minorNeighbors := tabula.SelectRowsWhere(&neighbors, in.ClassIdx,
		in.ClassMinor)

	return minorNeighbors
}

//
// randomGap return the neighbors gap between sample `p` and `n` using safe
// level (number of minority neighbors) of p in `lenslp` and `n` in `lensln`.
//
func (in *Runtime) randomGap(p, n tabula.Row, lenslp, lensln int) (
	delta float64,
) {
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
