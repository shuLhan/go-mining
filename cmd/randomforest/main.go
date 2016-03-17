// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/classifiers/randomforest"
	"github.com/shuLhan/tabula"
	"io/ioutil"
	"os"
	"strconv"
	"time"
)

var (
	// DEBUG level, can be set from environment variable.
	DEBUG = 0
	// nTree number of tree.
	nTree = 0
	// nRandomFeature number of feature to compute.
	nRandomFeature = 0
	// percentBoot percentage of sample for bootstraping.
	percentBoot = 0
)

var usage = func() {
	cmd := os.Args[0]
	fmt.Fprintf(os.Stderr, "Usage of %s:\n"+
		"[-ntree number] "+
		"[-nrandomfeature number] "+
		"[-percentboot number] "+
		"[config.dsv]\n", cmd)
	flag.PrintDefaults()
}

func init() {
	v := os.Getenv("DEBUG")
	if v == "" {
		DEBUG = 0
	} else {
		DEBUG, _ = strconv.Atoi(v)
	}

	flagUsage := []string{
		"Number of tree in forest (default 100)",
		"Number of feature to compute (default 0)",
		"Percentage of bootstrap (default 64%)",
	}

	flag.IntVar(&nTree, "ntree", -1, flagUsage[0])
	flag.IntVar(&nRandomFeature, "nrandomfeature", -1, flagUsage[1])
	flag.IntVar(&percentBoot, "percentboot", -1, flagUsage[2])
}

func trace(s string) (string, time.Time) {
	fmt.Println("[START]", s)
	return s, time.Now()
}

func un(s string, startTime time.Time) {
	endTime := time.Now()
	fmt.Println("[END]", s, "with elapsed time",
		endTime.Sub(startTime))
}

func createRandomForest(fcfg string) (*randomforest.Runtime, error) {
	rf := &randomforest.Runtime{}

	config, e := ioutil.ReadFile(fcfg)
	if e != nil {
		return nil, e
	}

	e = json.Unmarshal(config, rf)
	if e != nil {
		return nil, e
	}

	// Use option value from command parameter.
	if nTree > 0 {
		rf.NTree = nTree
	}
	if nRandomFeature > 0 {
		rf.NRandomFeature = nRandomFeature
	}
	if percentBoot > 0 {
		rf.PercentBoot = percentBoot
	}

	return rf, nil
}

func main() {
	defer un(trace("randomforest"))

	flag.Parse()

	if len(flag.Args()) <= 0 {
		usage()
		os.Exit(1)
	}

	fcfg := flag.Arg(0)

	// Parsing config file.
	rf, e := createRandomForest(fcfg)
	if e != nil {
		panic(e)
	}

	// Get dataset
	dataset := tabula.Claset{}
	_, e = dsv.SimpleRead(fcfg, &dataset)
	if e != nil {
		panic(e)
	}

	e = rf.Build(&dataset)
	if e != nil {
		panic(e)
	}

	fmt.Println("[randomforest] OOB mean value:", rf.OobErrMeanVal)
	fmt.Println("[randomforest] OOB oob steps:", rf.OobErrSteps)
}
