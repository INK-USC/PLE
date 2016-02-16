#!/bin/sh
Data='BBN'
Task='reduce_label_noise'
Prune='min'
Indir='data/' + $Data + '/' + $Task
MinOut=$Indir + '/min'
Results=$MinOut + '/Results'
RawJsonTrain=$Indir + '/train.json'
RawJsonTest=$Indir + '/test.json'
JsonTrain=$Indir + '/train_new.json'
JsonTest=$Indir + '/test_new.json'
BrownFile=$Indir + '/brown_bbn.txt'
FeatureFile=$Indir + '/feature.txt'
TypeFile=$Indir + '/type.txt'
TypeTypeFile=$Indir + '/type_type_kb.txt'
SuperTypeFile=$Indir + '/supertype.txt'
DistributionFile=$Indir + '/distribution_per_doc.txt'


# Make output dirs
mkdir -pv $Indir
mkdir $Indir+'/'+$Prune
mkdir $Restuls

##################################################
### Data Preparation

# NLP parse and generate features
python DataProcessor/nlp_parse.py $RawJsonTrain $JsonTrain
python DataProcessor/nlp_parse.py $RawJsonTest $JsonTest
python DataProcessor/ner_feature.py $JsonTrain $JsonTest $BrownFile $Indir
python DataProcessor/statistic.py $Indir

# Perform pruning
a=($(wc $FeatureFile))
FeatureNumber=${a[0]}
a=($(wc $TypeFile))
TypeNumber=${a[0]}
python DataProcessor/prune_heuristics.py $Data $MinOut $Prune $FeatureNumber $TypeNumber

# Generate type-type correlations from KB
python DataProcessor/type_type_kb.py -INDIR

# Copy needed files to #Prune subfolder
cp $FeatureFile $MinOut
cp $TypeFile $MinOut
cp $TypeTypeFile $MinOut
cp $SuperTypeFile $MinOut
cp $DistributionFile $MinOut

##################################################
### Train and Clean

# Train
Model/Embedding/ple/hple-corrKB -data $Data -task $Task+'/'+$Prune -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.6 -alpha 0.0001

# Clean the labels
Evaluation/emb_prediction.py $Data $Task+'/'+$Prune predict hple hete_corrKB topdown dot -100





