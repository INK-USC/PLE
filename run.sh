#!/bin/sh
Data='BBN'
Indir='Data/' + $Data
Intermediate='Intermediate/' + $Data
Outdir='Results/' + $Data
### Make intermediate and output dirs
mkdir -pv $Intermediate
mkdir -pv $Results

### Generate features
DataProcessor/feature_generation.py $Data

### Train PLE
Model/ple/hple-corrKB -data $Data -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.6 -alpha 0.0001

### Clean training labels
Evaluation/emb_prediction.py $Data hple hete_corrKB topdown dot -100

### Use denoised training set to train a type Classifier
Classifier/Classifier.py perceptron $Data hple hete_corrKB 0.003 20





