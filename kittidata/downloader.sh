#!/bin/bash

python3 checkdownloaded.py
aria2c -i "resturl.txt" -j4
