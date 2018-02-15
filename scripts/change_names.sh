#!/usr/bin/env bash

i=0
for path in $1/*; do
    [ -d "${path}" ] || continue
    dirname="$(basename "${path}")"
    i=$((i+1))
    mv ${dirname} "Uppgift_${i}"
done