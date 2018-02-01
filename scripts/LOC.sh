#!/usr/bin/env bash

cd ../data/bw
mkdir -p java

cd snippets

# From https://unix.stackexchange.com/questions/19654/changing-extension-to-multiple-files
for f in *.jsnp; do
cp -- "$f" "../java/${f%.jsnp}.java"
done

cd ../

out_dir="reports/LOC"
mkdir -p "${out_dir}"

current_date=$(date +"%Y-%m-%d_%H:%M:%S")
out_file="${out_dir}/loc.csv"

cloc --csv --report-file=${out_file} --by-file java/

rm -r java