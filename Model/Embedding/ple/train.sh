### dir: /srv/data/wenqihe/Experiments/Model/Embedding/hple/
echo 'data = ' $1 #
# echo 'lr = ' $1 # 0.25
# echo 'alpha = ' $2 # 0.0001
# echo 'iters = ' 80

### hple-noCorr
./hple-noCorr -data $1 -task reduce_label_noise/no -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.6 -alpha 0.0001

# ### hple-corrKB
./hple-corrKB -data $1 -task reduce_label_noise/no -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.6 -alpha 0.0001

### hple-corrH
./hple-corrH -data $1 -task reduce_label_noise/no -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.6 -alpha 0.0001
