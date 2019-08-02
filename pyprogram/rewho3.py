#!/usr/bin/env python

import os;
import re

os.open("who", 'r');

with os.open("who", 'r', 777) as f:
    for eachLine in f:
        print(re.split(r"\s\s+|\t", eachLine.strip()))

