#!/usr/bin/env bash

~/program/SourceMeter/Java/SourceMeterJava \
    -projectName=snippets \
    -runAndroidHunter=false \
    -runMetricHunter=false \
    -runVulnerabilityHunter=false \
    -runFaultHunter=false \
    -runRTEHunter=false \
    -runDCF=false \
    -resultsDir=../data/bw/reports/sourceMeter \
    -projectBaseDir=../data/bw/java