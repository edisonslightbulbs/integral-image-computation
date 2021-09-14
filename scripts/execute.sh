#!/usr/bin/env bash

# execute.sh:
#   executes project
#
# author: Everett
# created: 2021-09-14 07:29
# Github: https://github.com/antiqueeverett/

PROJECT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
echo "-- executing project"

# -- BIN directory
cd "$PROJECT_DIR" || return

mkdir -p ./output
rm -rf ./output/*.*
./build/bin/iimage --logtostderr=1
