// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dataset_test

import (
	"fmt"
	"testing"

	"github.com/shuLhan/go-mining/dataset"
)

var discV = []string{"a","b","b","a","a"}
var contV = []float64{0.2,0.4,0.6,0.8,1.0}

var attrs = []dataset.Attr{
	{false, &discV, []string{"a","b"}},
	{true, &contV, nil},
}

func TestGetContinuValue(t *testing.T) {
	exp := fmt.Sprint(contV)
	got := fmt.Sprint(*(attrs[1].GetContinuValues()))

	if exp != got {
		t.Fatal("Expecting:", exp,", got ", got)
	}
}
