// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package smote_test

import (
	"fmt"
	"io"
	"testing"

	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/knn"
	"github.com/shuLhan/go-mining/resampling/smote"
)

const (
	fcfg = "testdata/phoneme.dsv"
)

func doSmote (rw *dsv.ReadWriter) (e error) {
	var smote = &smote.SMOTE {
		Input		: knn.Input {
			Dataset		: nil,
			DistanceMethod	: knn.TEuclidianDistance,
			ClassIdx	: 5,
			K		: 5,
		},
		PercentOver	: 200,
		Synthetic	: nil,
	}

	classes := rw.Records.GroupByValue (smote.ClassIdx)
	minClass := classes.GetMinority ()

	fmt.Println ("minority samples:", minClass.Len ())

	smote.Dataset = minClass
	synthetic, e := smote.Resampling ()

	if e != nil {
		return
	}

	fmt.Println ("Synthetic:", synthetic.Len ())

	return
}

func TestSmote (t *testing.T) {
	var e error
	var n int
	var rw *dsv.ReadWriter

	rw = dsv.New ()

	e = rw.Open (fcfg)

	if nil != e {
		t.Fatal (e)
	}

	n, e = rw.Read ()

	if nil != e && e != io.EOF {
		t.Fatal (e)
	}

	fmt.Println ("Total samples:", n)

	e = doSmote (rw)

	if e != nil {
		t.Fatal (e)
	}

	rw.Close ()
}
