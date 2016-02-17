#!/bin/sh
Data='BBN'
Task='reduce_label_noise'
Indir='data/' + $Data + '/' + $Task

### Make output dirs
mkdir -pv $Indir+'/no/Results/'

### Generate features
DataProcessor/feature_generation.py $Data $Task

### Train 
Model/Embedding/ple/hple-corrKB -data $Data -task $Task -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.6 -alpha 0.0001

### Clean the labels
Evaluation/emb_prediction.py $Data $Task predict hple hete_corrKB topdown dot -100





