// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package classifiers

/*
Stats define list of statistic values.
*/
type Stats []Stat

//
// StartTimes return all start times in unix timestamp.
//
func (stats *Stats) StartTimes() (times []int64) {
	for _, stat := range *stats {
		times = append(times, stat.StartTime.Unix())
	}
	return
}

//
// EndTimes return all end times in unix timestamp.
//
func (stats *Stats) EndTimes() (times []int64) {
	for _, stat := range *stats {
		times = append(times, stat.EndTime.Unix())
	}
	return
}

/*
TPRates return all true-positive rate values.
*/
func (stats *Stats) TPRates() (tprates []float64) {
	for _, stat := range *stats {
		tprates = append(tprates, stat.TPRate)
	}
	return
}

/*
FPRates return all false-positive rate values.
*/
func (stats *Stats) FPRates() (fprates []float64) {
	for _, stat := range *stats {
		fprates = append(fprates, stat.FPRate)
	}
	return
}

/*
Precisions return all precision values.
*/
func (stats *Stats) Precisions() (precs []float64) {
	for _, stat := range *stats {
		precs = append(precs, stat.Precision)
	}
	return
}

/*
Recalls return all recall values.
*/
func (stats *Stats) Recalls() (recalls []float64) {
	return stats.TPRates()
}

/*
FMeasures return all F-measure values.
*/
func (stats *Stats) FMeasures() (fmeasures []float64) {
	for _, stat := range *stats {
		fmeasures = append(fmeasures, stat.FMeasure)
	}
	return
}

//
// Accuracies return all accuracy values.
//
func (stats *Stats) Accuracies() (accuracies []float64) {
	for _, stat := range *stats {
		accuracies = append(accuracies, stat.Accuracy)
	}
	return
}
