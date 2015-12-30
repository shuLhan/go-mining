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

/*
Input data for building CART.
*/
type Input struct {
	// SplitMethod define the criteria to used for splitting.
	SplitMethod int
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
		attrSubV := attrPartV.(set.SubsetString)
		splitV = attrSubV[0]
	}

	glog.V(2).Infoln(">>> maxgainindex:", MaxGainIdx)
	glog.V(2).Infoln(">>> split v:", splitV)

	node.Value = NodeValue{
		SplitAttrName:D.Columns[MaxGainIdx].GetName(),
		IsLeaf:       false,
		IsContinu:    MaxGain.IsContinu,
		Size:         nrow,
		SplitAttrIdx: MaxGainIdx,
		SplitV:       splitV,
	}

	splitL, splitR, e := D.SplitRowsByValue(MaxGainIdx, splitV)

	if e != nil {
		return node, e
	}

	// Set the Skip flag to true in attribute referenced by
	// MaxGainIdx, so it will not computed again in the next round.
	for i := range splitL.InputMetadata {
		if i == MaxGainIdx {
			splitL.InputMetadata[i].Skip = true
		} else {
			splitL.InputMetadata[i].Skip = false
		}
	}
	for i := range splitR.InputMetadata {
		if i == MaxGainIdx {
			splitR.InputMetadata[i].Skip = true
		} else {
			splitR.InputMetadata[i].Skip = false
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

/*
computeGiniGain calculate the gini index for each value in each attribute.
*/
func (in *Input) computeGiniGain(D *dataset.Reader) (gains []gini.Gini) {
	switch in.SplitMethod {
	case SplitMethodGini:
		// create gains value for all attribute minus target class.
		gains = make([]gini.Gini, D.GetNColumn())
	}

	targetV := D.GetTarget().ToStringSlice()
	classes := D.GetClass()

	for i := range (*D).InputMetadata {
		// skip class attribute.
		if i == D.ClassMetadataIndex {
			continue
		}

		// skip attribute with Skip is true
		if D.InputMetadata[i].Skip {
			gains[i].Skip = true
			continue
		}

		target := make([]string, len(targetV))
		copy(target, targetV)

		// compute gain.
		if (*D).InputMetadata[i].IsContinu {
			col := (*D).Columns[i]
			attr := col.ToFloatSlice()

			gains[i].ComputeContinu(&attr, &target, &classes)
		} else {
			attr := (*D).Columns[i].ToStringSlice()
			attrV := (*D).InputMetadata[i].NominalValues

			glog.V(2).Infoln(">>> attr :", attr)
			glog.V(2).Infoln(">>> attrV:", attrV)

			gains[i].ComputeDiscrete(&attr, &attrV, &target,
				&classes)
		}

		glog.V(2).Infoln(">>> gain :", gains[i])
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
				splitV := nodev.SplitV.(set.SliceString)
				attrV := (*data).Columns[nodev.SplitAttrIdx].
					Records[i].String()

				if set.IsSliceStringContain(splitV, attrV) {
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
