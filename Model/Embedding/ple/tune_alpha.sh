#!/bin/sh
### dir: /srv/data/wenqihe/Experiments/Model/Embedding/hple_noCorr/


### hple_noCorr-hete
for var in 0.00000001 0.00000003 0.0000001 0.0000003 
do
    echo $var
    echo ' '
    ./hple-corrKB -data BBN -task reduce_label_noise/no -mode bcd -size 50 -negatives 10 -iters 80 -threads 20 -lr 0.4 -alpha $var
    python /srv/data/wenqihe/Experiments/Evaluation/emb_test.py BBN reduce_label_noise/no predict hple_corrKB hete_feature maximum dot -100
done
