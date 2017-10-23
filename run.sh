#!/bin/sh
Data='BBN'
Indir='Data/BBN'
Intermediate='Intermediate/BBN'
Outdir='Results/BBN'
### Make intermediate and output dirs
mkdir -pv $Intermediate
mkdir -pv $Outdir

### Generate features
echo 'Feature Generation...'
python DataProcessor/feature_generation.py $Data 4
echo ' '

### Train PLE
### 	- Wiki: -iters 50 -lr 0.25
### 	- OntoNotes: -iters 50 -lr 0.3
### 	- BBN: -iters 80 -lr 0.4
echo 'Heterogeneous Partial-Label Embedding...'
Model/ple/hple -data $Data -mode bcd -size 50 -negatives 10 -iters 80 -threads 30 -lr 0.4 -alpha 0.0001
echo ' '

### Clean training labels
### 	- Wiki: maximum dot -1.0
### 	- OntoNotes: topdown dot -1
### 	- BBN: maximum dot -100
echo 'Label Noise Reduction with learned embeddings...'
python Evaluation/emb_prediction.py $Data hple hete_feature maximum dot -100
echo ' '

### Train a type Classifier over the de-noised training data; predict on test data
### 	- Wiki: 0.003 1
### 	- OntoNotes: 0.003 20
### 	- BBN: 0.003 50
echo 'Train classifier and predict...'
python Classifier/Classifier.py perceptron $Data hple hete_feature 0.003 30
echo ' '

### Evalaute prediction results...
echo 'Evaluate on test data...'
python Evaluation/evaluation.py $Data hple hete_feature
