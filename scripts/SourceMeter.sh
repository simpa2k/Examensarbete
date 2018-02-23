#!/usr/bin/env bash

~/program/SourceMeter/Java/SourceMeterJava \
    -currentDate=$4 \
    -projectName=$3 \
    -runAndroidHunter=false \
    -runMetricHunter=false \
    -runVulnerabilityHunter=false \
    -runFaultHunter=false \
    -runRTEHunter=false \
    -runDCF=false \
    -resultsDir=$2 \
    -projectBaseDir=$1