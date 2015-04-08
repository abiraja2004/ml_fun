#!/usr/bin/python

# input: beginning and ending year (e.g. 1996 2011), output filename
# output: per feature, per value, per value histogram

import json
import os
import sys

from common import Label

def main():
    start_year, end_year, out_fn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

    counts = {}
    for fn in ( os.path.join("data", "training", "%d.tdata" % year) for year in range(start_year, end_year + 1) ):
        print fn
        for line in open(fn):
            td = json.loads(line.strip())
            label = td.pop("label")
            for fname, fval in td.iteritems():
                counts.setdefault(fname, {}).setdefault(fval, dict([ (k, 0) for k in Label.get_all() ]))[label] += 1

    json.dump(counts, open(out_fn, "w"))

if __name__ == "__main__":
    main()
