// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package cart implement the Classification and Regression Tree by Breiman, et al.
CART is binary decision tree.

	Breiman, Leo, et al. Classification and regression trees. CRC press,
	1984.

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
	// Gains contain the gain value with additional information depends on
	// split method.
	Gains []gini.Gini
	// AttrMaxGain contain the attribute with the maximum gain.
	AttrMaxGain int
}

/*
NewInput create new Input object.
*/
func NewInput(SplitMethod int) (*Input) {
	return &Input{
		SplitMethod: SplitMethod,
		Tree: binary.Tree{},
	}
}

/*
BuildTree will create a tree using CART algorithm.
*/
func (in *Input) BuildTree(D *dataset.Input) (e error) {
	in.Tree.Root = in.splitTree(D)

	return e
}

/*
splitTree calculate the gain in all dataset, and split into two node: left and
right.

Return node with the split information.
*/
func (in *Input) splitTree(D *dataset.Input) (node *binary.BTNode) {
	node = &binary.BTNode{}

	// if all dataset is in the same class, return node as leaf with class
	// is set to that class.
	if D.IsInSingleClass() {
		targetAttr := (*D).GetTargetAttrValues()
		node.Value = NodeValue{
				IsLeaf: true,
				Class: (*targetAttr)[0],
			}

		return node
	}

	// if dataset is empty return node labeled with majority classes in
	// dataset.
	if D.Size <= 0 {
		majorClass := D.GetMajorityClass()
		node.Value = NodeValue{
			IsLeaf: true,
			Class: majorClass,
		}
	}

	// calculate the Gini gain for each attribute.
	in.computeGiniGain(D)

	// get attribute with maximum Gini gain.
	in.findAttrMaxGain()

	// using the sorted index in MaxGain, sort all field in dataset
	D.SortByIndex(in.AttrMaxGain, in.Gains[in.AttrMaxGain].SortedIndex)

	// Now that we have attribute with max gain in AttrMaxGain, and their
	// gain dan partition value in Gains[AttrMaxGain] and
	// GetMaxPartValue(), we split the dataset based on type of max-gain
	// attribute.
	// If its continuous, split the attribute using numeric value.
	// If its discrete, split the attribute using subset (partition) of
	// nominal values.
	var splitV interface{}

	if in.Gains[in.AttrMaxGain].IsContinu {
		splitV = in.Gains[in.AttrMaxGain].GetMaxPartGainValue()
	} else {
		attrPartV := in.Gains[in.AttrMaxGain].GetMaxPartGainValue()
		attrSubV := attrPartV.(set.SubsetString)
		splitV = attrSubV[0]
	}

	node.Value = NodeValue{
			IsLeaf: false,
			IsContinu: in.Gains[in.AttrMaxGain].IsContinu,
			SplitV: splitV,
		}

	splitD := D.SplitByAttrValue(in.AttrMaxGain, splitV)

	nodeLeft := in.splitTree(splitD)
	nodeRight := in.splitTree(D)

	node.SetLeft(nodeLeft)
	node.SetRight(nodeRight)

	return node
}

/*
computeGiniGain calculate the gini index for each value in each attribute.
*/
func (in *Input) computeGiniGain(D *dataset.Input) {
	switch in.SplitMethod {
	case SplitMethodGini:
		// create gains value for all attribute minus target class.
		in.Gains = make([]gini.Gini, len(D.Attrs) - 1)
	}

	targetAttr := D.GetTargetAttr()
	targetValues := targetAttr.GetDiscreteValues()
	classes := targetAttr.NominalValues

	for i := range (*D).Attrs {
		// skip class attribute.
		if i == D.ClassIdx {
			continue
		}

		target := make([]string, len(*targetValues))
		copy(target, (*targetValues))

		// compute gain.
		if (*D).Attrs[i].IsContinu {
			attr := (*D).Attrs[i].GetContinuValues()

			in.Gains[i].ComputeContinu(attr, &target, &classes)
		} else {
			attr := (*D).Attrs[i].GetDiscreteValues()
			attrValues := (*D).Attrs[i].NominalValues

			in.Gains[i].ComputeDiscrete(attr, &attrValues, &target,
							&classes)
		}
	}
}

/*
findAttrMaxGain find the attribute and value that have the maximum gain.
*/
func (in *Input) findAttrMaxGain() {
	var gainValue = 0.0
	var maxGainValue = 0.0

	for i := range in.Gains {
		gainValue = in.Gains[i].GetMaxGainValue()
		if gainValue > maxGainValue {
			maxGainValue = gainValue
			in.AttrMaxGain = i
		}
	}
}

/*
GetAttrMaxGain return the maximum gain in attribute.
*/
func (in *Input) GetAttrMaxGain() (*gini.Gini) {
	return &in.Gains[in.AttrMaxGain]
}

/*
String yes, it will print it JSON like format.
*/
func (in *Input) String() (s string) {
	s = in.Tree.String()
	return s
}
