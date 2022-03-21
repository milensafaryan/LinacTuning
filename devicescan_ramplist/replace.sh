#!/bin/bash

sed -i '' 's/L://g' "$1"
sed -i '' 's/(S)/_S/g' "$1"
sed -i '' 's/(R)/_R/g' "$1"
sed -i -e '1s/^/# /' "$1"
