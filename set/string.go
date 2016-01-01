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

// Strings is for working with element of list with type is string.
// Each element of slice is in the form of ["a", ..., "n"]
type Strings []string

// ListStrings is for working with lists of set.
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
SinglePartitionStrings create a table from a set of string, where each elemen
in a set become a single set.

Input: [a,b,c]
output:
    [
        [[a],[b],[c]]
    ]
*/
func SinglePartitionStrings(set Strings) (table TableStrings) {
	list := make(ListStrings, len(set))

	for x, el := range set {
		list[x] = Strings{el}
	}

	table = append(table, list)
	return
}

/*
JoinStringToTable will append string `s` to each set in list.
For example, given string `s` and input table `[[["a"]["b"]["c"]]]`, the
output table will be,
[
	[["a","s"]["b"]    ["c"]],
	[["a"]    ["b","s"]["c"]],
	[["a"]    ["b"]    ["c","s"]]
]
*/
func JoinStringToTable(s string, tableIn, tableOut TableStrings) TableStrings {
	for _, list := range tableIn {
		for y := range list {
			newList := make(ListStrings, len(list))
			copy(newList, list)
			newList[y] = append(newList[y], s)
			tableOut = append(tableOut, newList)
		}
	}
	return tableOut
}

/*
createIndent will create n space indentation and return it.
*/
func createIndent(n int) (s string) {
	for i := 0; i < n; i++ {
		s += " "
	}
	return
}

/*
PartitioningTableStrings will group the set's element `orgseed` into non-empty
lists, in such a way that every element is included in one and only of the
lists.

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

Number of possible list can be computed using Stirling number of second kind.

For more information see,
- https://en.wikipedia.org/wiki/Partition_of_a_set
*/
func PartitioningTableStrings(orgseed Strings, k int) (table TableStrings) {
	n := len(orgseed)
	seed := make(Strings, n)
	copy(seed, orgseed)

	glog.V(1).Infof("%s PartitioningTableStrings(%v,%v)\n",
		createIndent(n), n, k)

	// if only one split return the set contain only seed as list.
	// input: {a,b,c},  output: {{a,b,c}}
	if k == 1 {
		list := make(ListStrings, 1)
		list[0] = seed

		table := make(TableStrings, 1)
		table[0] = list
		return table
	}

	// if number of element in set equal with number split, return the set
	// that contain each element in list.
	// input: {a,b,c},  output:= {{a},{b},{c}}
	if n == k {
		return SinglePartitionStrings(seed)
	}

	nlist := math.StirlingS2(n, k)

	glog.V(1).Infof("%s Number of list: %v", createIndent(n), nlist)

	table = make(TableStrings, 0)

	// take the first element
	el := seed[0]

	// remove the first element from set
	seed = append(seed[:0], seed[1:]...)

	glog.V(1).Infof("%s el: %s, seed:", createIndent(n), el, seed)

	// generate child list
	genTable := PartitioningTableStrings(seed, k)

	glog.V(1).Infof("%s genTable join: %v", createIndent(n), genTable)

	// join elemen with generated set
	table = JoinStringToTable(el, genTable, table)

	glog.V(1).Infof("%s join %s      : %v\n", createIndent(n), el, table)

	genTable = PartitioningTableStrings(seed, k-1)

	glog.V(1).Infof("%s genTable append :", createIndent(n), genTable)

	for subidx := range genTable {
		list := make(ListStrings, len(genTable[subidx]))
		copy(list, genTable[subidx])
		list = append(list, Strings{el})
		table = append(table, list)
	}

	glog.V(1).Infof("%s append %v      : %v\n", createIndent(n), el, table)

	return
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
IsListStringsEqual compare two list of slice of string without regard to
their order.

	{{"a"},{"b"}} == {{"b"},{"a"}} is true.

Return true if both contain the same list, false otherwise.
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

Return true if both set is contain the same list, false otherwise.
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
