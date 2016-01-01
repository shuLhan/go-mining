// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package set is a library for working with the set.

Set is a slice of slice of slice of a type. For example, the set of string
could be displayed in the form of,

	{
		{{"a"},{"b"},{"c"}},
		{{"a","b"},{"c"}},
		{{"a"},{"b","c"}},
		{{"a","c"},{"b"}},
		{{"a","b","c"}},
	}
*/
package set

import (
	"github.com/golang/glog"
	"github.com/shuLhan/go-mining/math"
)

// Strings is for working with element of subset with type is string.
// Each element of slice is in the form of ["a", ..., "n"]
type Strings []string

// ListStrings is for working with subsets of set.
// Each elemen of slice is in the form of [["a"],["b","c"],...]
type ListStrings []Strings

// TableStrings is for working with set of slice of string.
// Each elemen in set is in the form of
// [
//	[["a"],["b","c"],...],
//	[["x"],["y",z"],...]
// ]
type TableStrings []ListStrings

/*
PartitioningTableStrings will group the set's element `orgseed` into non-empty
subsets, in such a way that every element is included in one and only of the
subsets.

Given a list of element in `orgseed`, and number of partition `k`, return
the set of all group of all elements without duplication.
For example, the set {a,b,c} if partitioned into 2 group will result in set

	{
		{{a,b},{c}},
		{{a,c},{b}},
		{{a},{b,c}},
	}

if partitioned into 3 group (k=3) will result in,

	{
		{{a},{b},{c}},
	}

Number of possible subset can be computed using Stirling number of second kind.

For more information see,
- https://en.wikipedia.org/wiki/Partition_of_a_set
- https://en.wikipedia.org/wiki/Partition_of_a_set
*/
func PartitioningTableStrings(orgseed Strings, k int) (rset TableStrings) {
	var s string

	n := len(orgseed)
	seed := make(Strings, n)
	copy(seed, orgseed)

	if glog.V(1) {
		s = ""
		for i := 0; i < n; i++ {
			s += " "
		}
		glog.Infof("%s PartitioningTableStrings(%v,%v)\n", s, n, k)
	}

	// if only one split return the set contain only seed as subset.
	// input: {a,b,c},  output: {{a,b,c}}
	if k == 1 {
		subset := make(ListStrings, 1)
		subset[0] = seed

		rset := make(TableStrings, 1)
		rset[0] = subset
		return rset
	}

	// if number of element in set equal with number split, return the set
	// that contain each element in subset.
	// input: {a,b,c},  output:= {{a},{b},{c}}
	if n == k {
		subset := make(ListStrings, n)

		for el := range seed {
			subset[el] = Strings{seed[el]}
		}

		rset := make(TableStrings, 1)
		rset[0] = subset

		return rset
	}

	nsubset := math.StirlingS2(n, k)

	if glog.V(1) {
		s = ""
		for i := 0; i < n; i++ {
			s += " "
		}
		glog.Infoln(s, " Number of subset:", nsubset)
	}

	rset = make(TableStrings, 0)

	// take the first element
	el := seed[0]

	// remove the first element from set
	seed = append(seed[:0], seed[1:]...)

	if glog.V(1) {
		s = ""
		for i := 0; i < n; i++ {
			s += " "
		}
		glog.Infoln(s, " el:", el, " seed:", seed)
	}

	// generate child subset
	genset := PartitioningTableStrings(seed, k)

	if glog.V(1) {
		s = ""
		for i := 0; i < n; i++ {
			s += " "
		}
		glog.Infoln(s, " genset join :", genset)
	}

	// join elemen with generated set
	for sub := range genset {
		for s := range genset[sub] {
			subset := make(ListStrings, len(genset[sub]))
			copy(subset, genset[sub])
			subset[s] = append(subset[s], el)
			rset = append(rset, subset)
		}
	}

	if glog.V(1) {
		s = ""
		for i := 0; i < n; i++ {
			s += " "
		}
		glog.Infof("%s join %v      : %v\n", s, el, rset)
	}

	genset = PartitioningTableStrings(seed, k-1)

	if glog.V(1) {
		s = ""
		for i := 0; i < n; i++ {
			s += " "
		}
		glog.Infoln(s, " genset append :", genset)
	}

	for subidx := range genset {
		subset := make(ListStrings, len(genset[subidx]))
		copy(subset, genset[subidx])
		subset = append(subset, Strings{el})
		rset = append(rset, subset)
	}

	if glog.V(1) {
		s = ""
		for i := 0; i < n; i++ {
			s += " "
		}
		glog.Infof("%s append %v      : %v\n", s, el, rset)
	}

	return rset
}

/*
IsStringsEqual compare elements of two slice of string without regard to
their order

	{"a","b"} == {"b","a"} is true

Return true if each both slice have the same elements, false otherwise.
*/
func IsStringsEqual(a Strings, b Strings) bool {
	if len(a) != len(b) {
		return false
	}

	check := make([]bool, len(a))

	for i := range a {
		for j := range b {
			if a[i] == b[j] {
				check[i] = true
			}
		}
	}

	for c := range check {
		if !check[c] {
			return false
		}
	}
	return true
}

/*
IsListStringsEqual compare two subset of slice of string without regard to
their order.

	{{"a"},{"b"}} == {{"b"},{"a"}} is true.

Return true if both contain the same subset, false otherwise.
*/
func IsListStringsEqual(a ListStrings, b ListStrings) bool {
	if len(a) != len(b) {
		return false
	}

	check := make([]bool, len(a))

	for i := range a {
		for j := range b {
			if IsStringsEqual(a[i], b[j]) {
				check[i] = true
				break
			}
		}
	}

	for c := range check {
		if !check[c] {
			return false
		}
	}
	return true
}

/*
IsTableStringsEqual compare two set of string without regard to their order.

	{
		{{"a"},{"b"}},
		{{"c"}}
	}

is equal to

	{
		{{"c"}},
		{{"b"},{"a"}}
	}

Return true if both set is contain the same subset, false otherwise.
*/
func IsTableStringsEqual(a TableStrings, b TableStrings) bool {
	if len(a) != len(b) {
		return false
	}

	check := make([]bool, len(a))

	for i := range a {
		for j := range b {
			if IsListStringsEqual(a[i], b[j]) {
				check[i] = true
				break
			}
		}
	}

	for c := range check {
		if !check[c] {
			return false
		}
	}
	return true
}

/*
IsStringsContain return true if slice of string contain elemen `el`,
otherwise return false.
*/
func IsStringsContain(ss Strings, el string) bool {
	for i := range ss {
		if ss[i] == el {
			return true
		}
	}
	return false
}
