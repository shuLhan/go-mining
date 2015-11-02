package set_test

import (
	"fmt"
	"testing"
	"github.com/shuLhan/go-mining/set"
)

var exp = set.SetString{
		{{"a"}, {"b","c"}},
		{{"b"}, {"a","c"}},
		{{"c"}, {"a","b"}},
	}

var b bool
var subset set.SubsetString
var setstr set.SetString

func TestIsSliceStringEqual(t *testing.T) {
	test := set.SubsetString {
		{"c","b"},
		{"c","a"},
	}
	check := []bool{ true, false }

	for i := range test {
		b := set.IsSliceStringEqual(exp[0][1], test[i])

		if b != check[i] {
			t.Fatal(exp[0][1]," == ", test[i], "? ", b)
		}
	}
}

func TestIsSubsetStringEqual(t *testing.T) {
	b = set.IsSubsetStringEqual(exp[0], exp[0])
	if ! b {
		t.Fatal("Expecting true, got", exp[0]," == ", exp[0], "? ", b)
	}

	subset = set.SubsetString{{"a"},{"c","b"},}
	b = set.IsSubsetStringEqual(exp[0], subset)
	if ! b {
		t.Fatal("Expecting true, got", exp[0]," == ", subset, "? ", b)
	}

	subset = set.SubsetString{{"a"},{"b","a"},}
	b = set.IsSubsetStringEqual(exp[0], subset)
	if b {
		t.Fatal("Expecting false, got", exp[0]," == ", subset, "? ", b)
	}

	b = set.IsSubsetStringEqual(exp[0], exp[1])
	if b {
		t.Fatal("Expecting false, got", exp[0]," == ", exp[1], "? ", b)
	}
}

func TestIsSetStringEqual(t *testing.T) {
	b = set.IsSetStringEqual(exp, exp)
	if ! b {
		t.Fatal("Expecting true, got", exp," == ", exp, "? ", b)
	}

	setstr = set.SetString{
			{{"c"}, {"a","b"}},
			{{"a"}, {"b","c"}},
			{{"b"}, {"a","c"}},
		}

	b = set.IsSetStringEqual(exp, setstr)
	if ! b {
		t.Fatal("Expecting true, got", exp," == ", setstr, "? ", b)
	}

	setstr = set.SetString{
			{{"c"}, {"a","b"}},
			{{"a"}, {"b","c"}},
		}

	b = set.IsSetStringEqual(exp, setstr)
	if b {
		t.Fatal("Expecting false, got", exp," == ", setstr, "? ", b)
	}

	setstr = set.SetString{
			{{"b"}, {"a","b"}},
			{{"c"}, {"a","b"}},
			{{"a"}, {"b","c"}},
		}

	b = set.IsSetStringEqual(exp, setstr)
	if b {
		t.Fatal("Expecting false, got", exp," == ", setstr, "? ", b)
	}
}

func TestPartitioningSetString(t *testing.T) {
	in := set.SubsetString{
			{"a","b","c"},
		}
	exp := []set.SetString{{
			{{"a","b","c"},},
		},{
			{{"a","b"},{"c"}},
			{{"b"},{"a","c"}},
			{{"a"},{"b","c"}},
		},{
			{{"a"},{"b"},{"c"}},
		},
	}
	split := []int{ 1, 2, 3 }

	for n := range in {
		for k := range split {
			fmt.Println("input:", in[n])

			setstr = set.PartitioningSetString(in[n], split[k])

			fmt.Println("result:", setstr)

			b = set.IsSetStringEqual(exp[k], setstr)
			if !b {
				t.Fatal("Expecting ", exp[k], " == ",
					setstr, "? ", b)
			}
		}
	}
}

func TestPartitioningSetString2(t *testing.T) {
	in := set.SubsetString{
			{"a","b","c",},
			{"a","b","c","d"},
			{"a","b","c","d","e"},
			{"a","b","c","d","e","f"},
		}

	for n := range in {
		for i := 1; i <= len(in[n]); i++ {
			fmt.Println("input:", in[n], ", k:", i)

			rset := set.PartitioningSetString(in[n], i)

			fmt.Println("result:", rset, ", n:", len(rset))
			fmt.Println()
		}
	}
}
