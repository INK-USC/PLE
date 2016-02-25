#!/bin/sh
Data='BBN'
Indir='Data/BBN'
Intermediate='Intermediate/BBN'
Outdir='Results/BBN'
### Make intermediate and output dirs
mkdir -pv $Intermediate
mkdir -pv $Outdir

### Generate features
echo 'Step 1 Generate Features'
python DataProcessor/feature_generation.py $Data
echo ' '

### Train PLE
echo 'Step 2 Train PLE'
Model/ple/hple-corrKB -data $Data -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.6 -alpha 0.0001
echo ' '

### Clean training labels
echo 'Step 3 Clean training labels'
python Evaluation/emb_prediction.py $Data hple_corrKB hete_feature topdown dot -100
echo ' '

### Use denoised training set to train a type Classifier
echo 'Step 4 Build a type classifier'
python Classifier/Classifier.py perceptron $Data hple_corrKB hete_feature 0.003 20




