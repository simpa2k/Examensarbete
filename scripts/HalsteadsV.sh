#!/usr/bin/env bash

cd ../data/bw
mkdir -p java

cd snippets

# From https://unix.stackexchange.com/questions/19654/changing-extension-to-multiple-files
for f in *.jsnp; do
    cp -- "$f" "../java/${f%.jsnp}.java"
done

cd ../
mkdir -p reports/Halstead

current_date=$(date +"%Y-%m-%d_%H:%M:%S")
out_dir="reports/Halstead"
out_file="${out_dir}/halstead.csv"

# java -cp ../../tools/:../../tools/HalsteadMetricsCMD.jar:../../tools/halsteadmetrics-1.0-SNAPSHOT-jar-with-dependencies.jar com.simonolofsson.writer.CSVWriter java ${out_file}
pwd
java -jar ../../tools/HalsteadMetricsCMD.jar java/ ${out_file} html

#rm -r java