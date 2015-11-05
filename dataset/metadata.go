// Copyright 2015 Mhd Sulhan <ms@kilabit.info>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dataset

import (
	"encoding/json"
	"log"

	"github.com/shuLhan/dsv"
)

/*
Metadata extension for common dataset.
*/
type Metadata struct {
	// dsv.Metadata as our base
	dsv.Metadata
	// IsContinu indicated whether the data continu or not.
	IsContinu	bool		`json:"IsContinu"`
	// NominalValues contain list of known discrete values in data.
	NominalValues	[]string	`json:"NominalValues"`
}

/*
String yes, it will print it JSON like format.
*/
func (md *Metadata) String() string {
	r, e := json.MarshalIndent (md, "", "\t")
	if nil != e {
		log.Print (e)
	}
	return string (r)
}
