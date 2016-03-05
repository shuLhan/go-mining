// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package cart implement the Classification and Regression Tree by Breiman, et al.
CART is binary decision tree.

	Breiman, Leo, et al. Classification and regression trees. CRC press,
	1984.

The implementation is based on Data Mining book,

	Han, Jiawei, Micheline Kamber, and Jian Pei. Data mining: concepts and
	techniques: concepts and techniques. Elsevier, 2011.
*/
package cart

import (
	"fmt"
	"github.com/shuLhan/go-mining/dataset"
	"github.com/shuLhan/go-mining/gain/gini"
	"github.com/shuLhan/go-mining/tree/binary"
	"github.com/shuLhan/tabula"
	"github.com/shuLhan/tabula/util"
	"github.com/shuLhan/tekstus"
	"os"
	"strconv"
)

const (
	// SplitMethodGini if defined in Input, the dataset will be splitted
	// using Gini gain for each possible value or partition.
	//
	// This option is used in Input.SplitMethod.
	SplitMethodGini = 0
)

const (
	// ColFlagParent denote that the column is parent/split node.
	ColFlagParent = 1
	// ColFlagSkip denote that the column would be skipped.
	ColFlagSkip = 2
)

var (
	// CART_DEBUG level, set from environment.
	CART_DEBUG = 0
)

/*
Input data for building CART.
*/
type Input struct {
	// SplitMethod define the criteria to used for splitting.
	SplitMethod int
	// NRandomFeature if less or equal to zero compute gain on all feature,
	// otherwise select n random feature and compute gain only on selected
	// features.
	NRandomFeature int
	// OOBErrVal is the last out-of-bag error value in the tree.
	OOBErrVal float64
	// Tree in classification.
	Tree binary.Tree
}

func init() {
	v := os.Getenv("CART_DEBUG")
	if v == "" {
		CART_DEBUG = 0
	} else {
		CART_DEBUG, _ = strconv.Atoi(v)
	}
}

/*
New create new Input object.
*/
func New(splitMethod, nRandomFeature int) *Input {
	return &Input{
		SplitMethod:    splitMethod,
		NRandomFeature: nRandomFeature,
		Tree:           binary.Tree{},
	}
}

/*
Build will create a tree using CART algorithm.
*/
func (in *Input) Build(D *dataset.Reader) (e error) {
	in.Tree.Root, e = in.splitTreeByGain(D)

	return
}

/*
splitTreeByGain calculate the gain in all dataset, and split into two node:
left and right.

Return node with the split information.
*/
func (in *Input) splitTreeByGain(D *dataset.Reader) (node *binary.BTNode,
	e error,
) {
	node = &binary.BTNode{}

	// if dataset is empty return node labeled with majority classes in
	// dataset.
	nrow := D.GetNRow()

	if nrow <= 0 {
		if CART_DEBUG >= 2 {
			fmt.Printf("[cart] empty dataset (%s) : %v\n",
				D.GetMajorityClass(), D)
		}

		node.Value = NodeValue{
			IsLeaf: true,
			Class:  D.GetMajorityClass(),
			Size:   0,
		}
		return node, nil
	}

	// if all dataset is in the same class, return node as leaf with class
	// is set to that class.
	single, name := D.IsInSingleClass()
	if single {
		if CART_DEBUG >= 2 {
			fmt.Printf("[cart] in single class (%s): %v\n", name,
				D.Columns)
		}

		node.Value = NodeValue{
			IsLeaf: true,
			Class:  name,
			Size:   nrow,
		}
		return node, nil
	}

	if CART_DEBUG >= 2 {
		fmt.Println("[cart] D:", D)
	}

	// calculate the Gini gain for each attribute.
	gains := in.computeGiniGain(D)

	// get attribute with maximum Gini gain.
	MaxGainIdx := gini.FindMaxGain(&gains)
	MaxGain := gains[MaxGainIdx]

	// if maxgain value is 0, use majority class as node and terminate
	// the process
	if MaxGain.GetMaxGainValue() == 0 {
		if CART_DEBUG >= 2 {
			fmt.Println("[cart] max gain 0 with target",
				D.GetTarget(), " and majority class is ",
				D.GetMajorityClass())
		}

		node.Value = NodeValue{
			IsLeaf: true,
			Class:  D.GetMajorityClass(),
			Size:   0,
		}
		return node, nil
	}

	// using the sorted index in MaxGain, sort all field in dataset
	D.SortColumnsByIndex(MaxGain.SortedIndex)

	if CART_DEBUG >= 2 {
		fmt.Println("[cart] maxgain:", MaxGain)
	}

	// Now that we have attribute with max gain in MaxGainIdx, and their
	// gain dan partition value in Gains[MaxGainIdx] and
	// GetMaxPartValue(), we split the dataset based on type of max-gain
	// attribute.
	// If its continuous, split the attribute using numeric value.
	// If its discrete, split the attribute using subset (partition) of
	// nominal values.
	var splitV interface{}

	if MaxGain.IsContinu {
		splitV = MaxGain.GetMaxPartGainValue()
	} else {
		attrPartV := MaxGain.GetMaxPartGainValue()
		attrSubV := attrPartV.(tekstus.ListStrings)
		splitV = attrSubV[0]
	}

	if CART_DEBUG >= 2 {
		fmt.Println("[cart] maxgainindex:", MaxGainIdx)
		fmt.Println("[cart] split v:", splitV)
	}

	node.Value = NodeValue{
		SplitAttrName: D.Columns[MaxGainIdx].GetName(),
		IsLeaf:        false,
		IsContinu:     MaxGain.IsContinu,
		Size:          nrow,
		SplitAttrIdx:  MaxGainIdx,
		SplitV:        splitV,
	}

	splitL, splitR, e := D.SplitRowsByValue(MaxGainIdx, splitV)

	if e != nil {
		return node, e
	}

	// Set the flag to parent in attribute referenced by
	// MaxGainIdx, so it will not computed again in the next round.
	for x := range splitL.Columns {
		if x == MaxGainIdx {
			splitL.Columns[x].Flag = ColFlagParent
		} else {
			splitL.Columns[x].Flag = 0
		}
	}
	for x := range splitR.Columns {
		if x == MaxGainIdx {
			splitR.Columns[x].Flag = ColFlagParent
		} else {
			splitR.Columns[x].Flag = 0
		}
	}

	nodeLeft, e := in.splitTreeByGain(splitL)
	if e != nil {
		return node, e
	}

	nodeRight, e := in.splitTreeByGain(splitR)
	if e != nil {
		return node, e
	}

	node.SetLeft(nodeLeft)
	node.SetRight(nodeRight)

	return node, nil
}

// SelectRandomFeature if NRandomFeature is greater than zero, select and
// compute gain in n random features instead of in all features
func (in *Input) SelectRandomFeature(D *dataset.Reader) {
	if in.NRandomFeature <= 0 {
		// all features selected
		return
	}
	// count all features minus class
	nfeature := D.GetNColumn() - 1
	if in.NRandomFeature >= nfeature {
		// number of random feature equal or greater than number of
		// feature in dataset
		return
	}

	// exclude class index and parent node index
	excludeIdx := []int{D.ClassIndex}
	for x, col := range (*D).Columns {
		if (col.Flag & ColFlagParent) == ColFlagParent {
			excludeIdx = append(excludeIdx, x)
		} else {
			D.Columns[x].Flag |= ColFlagSkip
		}
	}

	var pickedIdx []int
	for x := 0; x < in.NRandomFeature; x++ {
		idx := util.GetRandomInteger(D.GetNColumn(), false, pickedIdx,
			excludeIdx)
		pickedIdx = append(pickedIdx, idx)

		// Remove skip flag on selected column
		D.Columns[idx].Flag = D.Columns[idx].Flag &^ ColFlagSkip
	}

	if CART_DEBUG >= 1 {
		fmt.Println("[cart] selected random features: ", pickedIdx)
		fmt.Println("[cart] selected columns        : ", D.Columns)
	}
}

/*
computeGiniGain calculate the gini index for each value in each attribute.
*/
func (in *Input) computeGiniGain(D *dataset.Reader) (gains []gini.Gini) {
	switch in.SplitMethod {
	case SplitMethodGini:
		// create gains value for all attribute minus target class.
		gains = make([]gini.Gini, D.GetNColumn())
	}

	in.SelectRandomFeature(D)

	targetV := D.GetTarget().ToStringSlice()
	classes := D.GetTargetClass()

	for x, col := range (*D).Columns {
		// skip class attribute.
		if x == D.ClassIndex {
			continue
		}

		// skip column flagged with parent
		if (col.Flag & ColFlagParent) == ColFlagParent {
			gains[x].Skip = true
			continue
		}

		// ignore column flagged with skip
		if (col.Flag & ColFlagSkip) == ColFlagSkip {
			gains[x].Skip = true
			continue
		}

		target := make([]string, len(targetV))
		copy(target, targetV)

		// compute gain.
		if col.GetType() == tabula.TReal {
			attr := col.ToFloatSlice()

			gains[x].ComputeContinu(&attr, &target, &classes)
		} else {
			attr := col.ToStringSlice()
			attrV := col.ValueSpace

			if CART_DEBUG >= 2 {
				fmt.Println("[cart] attr :", attr)
				fmt.Println("[cart] attrV:", attrV)
			}

			gains[x].ComputeDiscrete(&attr, &attrV, &target,
				&classes)
		}

		if CART_DEBUG >= 2 {
			fmt.Println("[cart] gain :", gains[x])
		}
	}
	return
}

/*
Classify return the prediction of one sample.
*/
func (in *Input) Classify(data tabula.Row) (class string) {
	node := in.Tree.Root
	nodev := node.Value.(NodeValue)

	for !nodev.IsLeaf {
		if nodev.IsContinu {
			splitV := nodev.SplitV.(float64)
			attrV := data[nodev.SplitAttrIdx].Float()

			if attrV < splitV {
				node = node.Left
			} else {
				node = node.Right
			}
		} else {
			splitV := nodev.SplitV.(tekstus.Strings)
			attrV := data[nodev.SplitAttrIdx].String()

			if splitV.IsContain(attrV) {
				node = node.Left
			} else {
				node = node.Right
			}
		}
		nodev = node.Value.(NodeValue)
	}

	return nodev.Class
}

/*
ClassifySet set the class attribute based on tree classification.
*/
func (in *Input) ClassifySet(data *dataset.Reader) (e error) {
	nrow := data.GetNRow()
	targetAttr := data.GetTarget()

	for i := 0; i < nrow; i++ {
		class := in.Classify(*(*data).GetRow(i))

		(*targetAttr).Records[i].V = class
	}

	return
}

/*
CountOOBError process out-of-bag data on tree and return error value.
*/
func (in *Input) CountOOBError(oob dataset.Reader) (errval float64, e error) {
	// save the original target to be compared later.
	origTarget := oob.GetTarget().ToStringSlice()

	if CART_DEBUG >= 2 {
		fmt.Println("[cart] OOB:", oob.Columns)
		fmt.Println("[cart] TREE:", &in.Tree)
	}

	// reset the target.
	oob.GetTarget().ClearValues()

	e = in.ClassifySet(&oob)

	if e != nil {
		// set original target values back.
		oob.GetTarget().SetValues(origTarget)
		return
	}

	target := oob.GetTarget().ToStringSlice()

	if CART_DEBUG >= 2 {
		fmt.Println("[cart] original target:", origTarget)
		fmt.Println("[cart] classify target:", target)
	}

	// count how many target value is miss-classified.
	in.OOBErrVal, _, _ = tekstus.WordsCountMissRate(origTarget, target)

	// set original target values back.
	oob.GetTarget().SetValues(origTarget)

	return in.OOBErrVal, nil
}

/*
String yes, it will print it JSON like format.
*/
func (in *Input) String() (s string) {
	s = in.Tree.String()
	return s
}
