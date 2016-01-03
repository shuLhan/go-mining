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
	"github.com/golang/glog"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/dsv/util"
	"github.com/shuLhan/go-mining/dataset"
	"github.com/shuLhan/go-mining/gain/gini"
	"github.com/shuLhan/go-mining/set"
	"github.com/shuLhan/go-mining/tree/binary"
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
	// tree in classification.
	Tree binary.Tree
}

/*
NewInput create new Input object.
*/
func NewInput(SplitMethod int) *Input {
	return &Input{
		SplitMethod: SplitMethod,
		Tree:        binary.Tree{},
	}
}

/*
BuildTree will create a tree using CART algorithm.
*/
func (in *Input) BuildTree(D *dataset.Reader) (e error) {
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
		node.Value = NodeValue{
			IsLeaf: true,
			Class:  name,
			Size:   nrow,
		}
		return node, nil
	}

	glog.V(2).Infoln(">>> D:", D)

	// calculate the Gini gain for each attribute.
	gains := in.computeGiniGain(D)

	// get attribute with maximum Gini gain.
	MaxGainIdx := gini.FindMaxGain(&gains)
	MaxGain := gains[MaxGainIdx]

	// if maxgain value is 0, use majority class as node and terminate
	// the process
	if MaxGain.GetMaxGainValue() == 0 {
		glog.V(2).Infoln(">>> max gain 0 with target",
			D.GetTarget(), " and majority class is ",
			D.GetMajorityClass())

		node.Value = NodeValue{
			IsLeaf: true,
			Class:  D.GetMajorityClass(),
			Size:   0,
		}
		return node, nil
	}

	// using the sorted index in MaxGain, sort all field in dataset
	D.SortColumnsByIndex(MaxGain.SortedIndex)

	glog.V(2).Infoln(">>> maxgain:", MaxGain)

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
		attrSubV := attrPartV.(set.ListStrings)
		splitV = attrSubV[0]
	}

	glog.V(2).Infoln(">>> maxgainindex:", MaxGainIdx)
	glog.V(2).Infoln(">>> split v:", splitV)

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
	for x, col := range splitL.Columns {
		if x == MaxGainIdx {
			col.Flag = ColFlagParent
		} else {
			col.Flag = 0
		}
	}
	for x, col := range splitR.Columns {
		if x == MaxGainIdx {
			col.Flag = ColFlagParent
		} else {
			col.Flag = 0
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

	// exclude class index and parent node index
	excludeIdx := []int{D.ClassIndex}
	for x, col := range (*D).Columns {
		if (col.Flag & ColFlagParent) == ColFlagParent {
			excludeIdx = append(excludeIdx, x)
		}
		D.Columns[x].Flag |= ColFlagSkip
	}

	var pickedIdx []int
	for x := 0; x < in.NRandomFeature; x++ {
		idx := util.GetRandomInteger(D.GetNColumn(), false, pickedIdx,
			excludeIdx)
		pickedIdx = append(pickedIdx, idx)

		D.Columns[idx].Flag = D.Columns[idx].Flag &^ ColFlagSkip
	}

	glog.V(1).Info(">>> selected random features: ", pickedIdx)
	glog.V(1).Info(">>> selected columns        : ", D.Columns)
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
		if col.GetType() == dsv.TReal {
			attr := col.ToFloatSlice()

			gains[x].ComputeContinu(&attr, &target, &classes)
		} else {
			attr := col.ToStringSlice()
			attrV := col.ValueSpace

			glog.V(2).Infoln(">>> attr :", attr)
			glog.V(2).Infoln(">>> attrV:", attrV)

			gains[x].ComputeDiscrete(&attr, &attrV, &target,
				&classes)
		}

		glog.V(2).Infoln(">>> gain :", gains[x])
	}
	return
}

/*
ClassifySet set the class attribute based on tree classification.
*/
func (in *Input) ClassifySet(data *dataset.Reader) (e error) {
	var node *binary.BTNode
	var nodev NodeValue

	nrow := data.GetNRow()
	targetAttr := data.GetTarget()

	for i := 0; i < nrow; i++ {
		node = in.Tree.Root
		nodev = node.Value.(NodeValue)

		for !nodev.IsLeaf {
			if nodev.IsContinu {
				splitV := nodev.SplitV.(float64)
				attrV := (*data).Columns[nodev.SplitAttrIdx].
					Records[i].Float()

				if attrV < splitV {
					node = node.Left
				} else {
					node = node.Right
				}
			} else {
				splitV := nodev.SplitV.(set.Strings)
				attrV := (*data).Columns[nodev.SplitAttrIdx].
					Records[i].String()

				if set.IsStringsContain(splitV, attrV) {
					node = node.Left
				} else {
					node = node.Right
				}
			}
			nodev = node.Value.(NodeValue)
		}

		(*targetAttr).Records[i].V = nodev.Class
	}

	return
}

/*
CountOOBError process out-of-bag data on tree and return error value.
*/
func (in *Input) CountOOBError(oob dataset.Reader) (errval float64, e error) {
	n := float64(oob.GetNRow())

	// save the original target to be compared later.
	origTarget := oob.GetTarget().ToStringSlice()

	glog.V(2).Info(">>> OOB:", oob.Columns)
	glog.V(2).Info(">>> TREE:", &in.Tree)

	// reset the target.
	oob.GetTarget().ClearValues()

	e = in.ClassifySet(&oob)

	if e != nil {
		// set original target values back.
		oob.GetTarget().SetValues(origTarget)
		return
	}

	// count how many target value is miss-classified.
	var miss float64

	target := oob.GetTarget().ToStringSlice()

	glog.V(2).Info(">>> original target:", origTarget)
	glog.V(2).Info(">>> classify target:", target)

	for x, row := range target {
		if row != origTarget[x] {
			glog.V(1).Info(">>> miss ", oob.Rows[x],
				" expecting ", origTarget[x])
			miss++
		}
	}

	in.OOBErrVal = float64(miss / n)

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
