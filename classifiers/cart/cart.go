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
	if D.GetNRow() <= 0 {
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
			Size:   D.GetNRow(),
		}

		return node, nil
	}

	// calculate the Gini gain for each attribute.
	gains := in.computeGiniGain(D)

	// get attribute with maximum Gini gain.
	MaxGainIdx := gini.FindMaxGain(&gains)
	MaxGain := gains[MaxGainIdx]

	// using the sorted index in MaxGain, sort all field in dataset
	D.SortColumnsByIndex(*MaxGain.SortedIndex)

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

	node.Value = NodeValue{
		IsLeaf:       false,
		IsContinu:    MaxGain.IsContinu,
		Size:         D.GetNRow(),
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
			continue
		}

		target := make([]string, len(targetV))
		copy(target, targetV)

		// compute gain.
		if (*D).InputMetadata[i].IsContinu {
			attr := (*D).Columns[i].ToFloatSlice()

			gains[i].ComputeContinu(&attr, &target, &classes)
		} else {
			attr := (*D).Columns[i].ToStringSlice()
			attrV := (*D).InputMetadata[i].NominalValues

			gains[i].ComputeDiscrete(&attr, &attrV, &target,
				&classes)
		}
	}
	return
}

/*
Classify set the class attribute based on tree classification.
*/
func (in *Input) ClassifySet(data *dataset.Reader) (e error) {
	var node *binary.BTNode
	var nodev NodeValue

	targetAttr := data.GetTarget()

	for i := 0; i < data.GetNRow(); i++ {
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
String yes, it will print it JSON like format.
*/
func (in *Input) String() (s string) {
	s = in.Tree.String()
	return s
}
