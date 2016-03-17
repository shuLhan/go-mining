// Copyright 2016 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/shuLhan/dsv"
	"github.com/shuLhan/go-mining/classifiers/cart"
	"github.com/shuLhan/tabula"
	"io/ioutil"
	"os"
	"strconv"
	"time"
)

var (
	// DEBUG level, can be set from environment variable.
	DEBUG          = 0
	nRandomFeature = 0
)

var usage = func() {
	cmd := os.Args[0]
	fmt.Fprintf(os.Stderr, "Usage of %s: [-n number] [config.dsv]\n", cmd)
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
		"Number of random feature (default 0)",
	}

	flag.IntVar(&nRandomFeature, "n", 0, flagUsage[0])
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

func createCart(fcfg string) (*cart.Runtime, error) {
	cartrt := &cart.Runtime{}

	config, e := ioutil.ReadFile(fcfg)
	if e != nil {
		return nil, e
	}

	e = json.Unmarshal(config, cartrt)
	if e != nil {
		return nil, e
	}

	if nRandomFeature > 0 {
		cartrt.NRandomFeature = nRandomFeature
	}

	return cartrt, nil
}

func main() {
	defer un(trace("cart"))

	flag.Parse()

	if len(flag.Args()) <= 0 {
		usage()
		os.Exit(1)
	}

	fcfg := flag.Arg(0)

	// Parsing config file and check command parameter values.
	cartrt, e := createCart(fcfg)
	if e != nil {
		panic(e)
	}

	// Get dataset
	dataset := tabula.Claset{}
	_, e = dsv.SimpleRead(fcfg, &dataset)
	if e != nil {
		panic(e)
	}

	if DEBUG >= 1 {
		fmt.Printf("[cart] Class index: %v\n", dataset.GetClassIndex())
	}

	e = cartrt.Build(&dataset)
	if e != nil {
		panic(e)
	}

	if DEBUG >= 1 {
		fmt.Println("[cart] CART tree:\n", cartrt)
	}
}
