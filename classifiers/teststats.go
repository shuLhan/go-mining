// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package classifiers

type TestStats struct {
	// NSample number of sample used for testing
	NSample int
	// NPositive number of positive result
	NPositive int
	// NNegative number of negative result
	NNegative int
}

func (tstats *TestStats) GetTPRate() float64 {
	return float64(tstats.NPositive) / float64(tstats.NSample)
}

func (tstats *TestStats) GetFPRate() float64 {
	return float64(tstats.NNegative) / float64(tstats.NSample)
}

func (tstats *TestStats) GetTNRate() float64 {
	return 1 - tstats.GetFPRate()
}

func (tstats *TestStats) GetRecall() float64 {
	return tstats.GetTPRate()
}
