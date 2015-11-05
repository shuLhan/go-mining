// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dataset_test

import (
	"fmt"
	"testing"
	"github.com/shuLhan/go-mining/dataset"
)

var discV2 = []string{"x","y","y","x","x"}
var contV2 = []float64{1,2,2,1,1}

func TestSplitByAttrValue(t *testing.T) {
	var attrs = []dataset.Attr{
		{false, &discV, []string{"a","b"}},
		{false, &discV2, []string{"x","y"}},
		{true, &contV, nil},
		{true, &contV2, nil},
	}
	var splitExps = []string{
		"[a a a]",
		"[x x x]",
		"[0.2 0.8 1]",
		"[1 1 1]",
	}
	var inputExps = []string{
		"[b b]",
		"[y y]",
		"[0.4 0.6]",
		"[2 2]",
	}

	input := dataset.Input{}

	input.Attrs = attrs

	split := input.SplitAttrDiscrete(0, []string{"a"})

	for i,a := range split.Attrs {
		got := fmt.Sprint(a.Values)

		if splitExps[i] != got {
			t.Fatal("Expecting ", splitExps[i]," got ", got)
		}

		got = fmt.Sprint(input.Attrs[i].Values)

		if inputExps[i] != got {
			t.Fatal("Expecting ", inputExps[i]," got ", got)
		}
	}
}

func TestSpliyByAttrValue2(t *testing.T) {
	var attrs = []dataset.Attr{
		{false, &discV, []string{"a","b"}},
		{false, &discV2, []string{"x","y"}},
		{true, &contV, nil},
		{true, &contV2, nil},
	}
	var inputExps = []string{
		"[a a a]",
		"[x x x]",
		"[0.2 0.8 1]",
		"[1 1 1]",
	}
	var splitExps = []string{
		"[b b]",
		"[y y]",
		"[0.4 0.6]",
		"[2 2]",
	}

	input := dataset.Input{}

	input.Attrs = attrs

	split := input.SplitAttrDiscrete(0, []string{"b"})

	for i,a := range split.Attrs {
		got := fmt.Sprint(a.Values)

		if splitExps[i] != got {
			t.Fatal("Expecting ", splitExps[i]," got ", got)
		}

		got = fmt.Sprint(input.Attrs[i].Values)

		if inputExps[i] != got {
			t.Fatal("Expecting ", inputExps[i]," got ", got)
		}
	}
}
