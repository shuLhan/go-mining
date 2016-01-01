// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package set_test

import (
	"fmt"
	"github.com/shuLhan/go-mining/set"
	"testing"
)

var exp = set.TableStrings{
	{{"a"}, {"b", "c"}},
	{{"b"}, {"a", "c"}},
	{{"c"}, {"a", "b"}},
}

var b bool
var subset set.ListStrings
var setstr set.TableStrings

func TestIsStringsEqual(t *testing.T) {
	test := set.ListStrings{
		{"c", "b"},
		{"c", "a"},
	}
	check := []bool{true, false}

	for i := range test {
		b := set.IsStringsEqual(exp[0][1], test[i])

		if b != check[i] {
			t.Fatal(exp[0][1], " == ", test[i], "? ", b)
		}
	}
}

func TestIsListStringsEqual(t *testing.T) {
	b = set.IsListStringsEqual(exp[0], exp[0])
	if !b {
		t.Fatal("Expecting true, got", exp[0], " == ", exp[0], "? ", b)
	}

	subset = set.ListStrings{{"a"}, {"c", "b"}}
	b = set.IsListStringsEqual(exp[0], subset)
	if !b {
		t.Fatal("Expecting true, got", exp[0], " == ", subset, "? ", b)
	}

	subset = set.ListStrings{{"a"}, {"b", "a"}}
	b = set.IsListStringsEqual(exp[0], subset)
	if b {
		t.Fatal("Expecting false, got", exp[0], " == ", subset, "? ", b)
	}

	b = set.IsListStringsEqual(exp[0], exp[1])
	if b {
		t.Fatal("Expecting false, got", exp[0], " == ", exp[1], "? ", b)
	}
}

func TestIsTableStringsEqual(t *testing.T) {
	b = set.IsTableStringsEqual(exp, exp)
	if !b {
		t.Fatal("Expecting true, got", exp, " == ", exp, "? ", b)
	}

	setstr = set.TableStrings{
		{{"c"}, {"a", "b"}},
		{{"a"}, {"b", "c"}},
		{{"b"}, {"a", "c"}},
	}

	b = set.IsTableStringsEqual(exp, setstr)
	if !b {
		t.Fatal("Expecting true, got", exp, " == ", setstr, "? ", b)
	}

	setstr = set.TableStrings{
		{{"c"}, {"a", "b"}},
		{{"a"}, {"b", "c"}},
	}

	b = set.IsTableStringsEqual(exp, setstr)
	if b {
		t.Fatal("Expecting false, got", exp, " == ", setstr, "? ", b)
	}

	setstr = set.TableStrings{
		{{"b"}, {"a", "b"}},
		{{"c"}, {"a", "b"}},
		{{"a"}, {"b", "c"}},
	}

	b = set.IsTableStringsEqual(exp, setstr)
	if b {
		t.Fatal("Expecting false, got", exp, " == ", setstr, "? ", b)
	}
}

func TestPartitioningTableStrings(t *testing.T) {
	in := set.ListStrings{
		{"a", "b", "c"},
	}
	exp := []set.TableStrings{{
		{{"a", "b", "c"}},
	}, {
		{{"a", "b"}, {"c"}},
		{{"b"}, {"a", "c"}},
		{{"a"}, {"b", "c"}},
	}, {
		{{"a"}, {"b"}, {"c"}},
	},
	}
	split := []int{1, 2, 3}

	for n := range in {
		for k := range split {
			fmt.Println("input:", in[n])

			setstr = set.PartitioningTableStrings(in[n], split[k])

			fmt.Println("result:", setstr)

			b = set.IsTableStringsEqual(exp[k], setstr)
			if !b {
				t.Fatal("Expecting ", exp[k], " == ",
					setstr, "? ", b)
			}
		}
	}
}

func TestPartitioningTableStrings2(t *testing.T) {
	in := set.ListStrings{
		{"a", "b", "c"},
		{"a", "b", "c", "d"},
		{"a", "b", "c", "d", "e"},
		{"a", "b", "c", "d", "e", "f"},
	}

	for n := range in {
		for i := 1; i <= len(in[n]); i++ {
			fmt.Println("input:", in[n], ", k:", i)

			rset := set.PartitioningTableStrings(in[n], i)

			fmt.Println("result:", rset, ", n:", len(rset))
			fmt.Println()
		}
	}
}
