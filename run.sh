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
python DataProcessor/feature_generation.py $Data 10
echo ' '

### Train PLE
echo 'Step 2 Heterogeneous Partial-Label Embedding'
Model/ple/hple -data $Data -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.6 -alpha 0.0001
echo ' '

### Clean training labels
echo 'Step 3 Label Noise Reduction with learned embeddings'
python Evaluation/emb_prediction.py $Data hple hete_feature topdown dot -100
echo ' '

### Train a type Classifier over the de-noised training data 
echo 'Step 4 Build a type classifier'
python Classifier/Classifier.py perceptron $Data hple hete_feature 0.003 20


