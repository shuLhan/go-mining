package knn_test

import (
	"fmt"
	"testing"

	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/knn"
)

func TestComputeEuclidianDistance (t *testing.T) {
	var exp = []string {
		`&[0.302891 0.608544 0.47413 1.42718 -0.811085 1]`,
		`&[&[0.1048 0.5756 0.3424 0.8304 0 1] 1.3306502169991932
 &[0.318095 0.810884 0.818231 0.820952 0.860136 1] 1.684961127148042
 &[0.540984 2.1451 -0.561563 -0.227764 -0.140565 1] 2.2662316739468626
 &[1.03941 2.27108 1.69893 -0.36917 -0.167506 1] 2.4624751775398668
 &[0.655083 1.0539 1.37163 -0.723877 1.77021 1] 2.535231744831229
]`,
	}

	var e error
	var instance *dsv.RecordSlice
	var kneighbors *knn.DistanceSlice
	var input = &knn.Input {}
	var dsvrw = dsv.New ()
	var classes dsv.MapStringRow

	e = dsvrw.Open ("phoneme.dsv")

	if nil != e {
		return
	}

	dsvrw.Read ()
	dsvrw.Close ()

	// Processing
	input.Method	= knn.TEuclidianDistance
	input.ClassIdx	= 5
	input.K		= 5
	classes		= dsvrw.Records.GroupByValue (input.K)
	input.Dataset	= classes.GetMinority ()

	instance = (input.Dataset.Front ()).Value.(*dsv.RecordSlice)

	if nil != e {
		t.Fatal (e)
	}

	kneighbors, e = input.Neighbors (instance)

	got	:= fmt.Sprint (instance)
	i	:= 0
	if got != exp[i] {
		t.Fatal ("Expecting:\n", exp[i], "\n Got:\n", got)
	}

	got	= fmt.Sprint (kneighbors)
	i	= 1
	if got != exp[i] {
		t.Fatal ("Expecting:\n", exp[i], "\n Got:\n", got)
	}
}
