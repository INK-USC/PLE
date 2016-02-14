### dir: /srv/data/wenqihe/Experiments/Model/Embedding/hple_noCorr/


### hple_noCorr-hete
./hple-noCorr -data task2 -task reduce_label_noise/10 -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.25 -alpha 0.0001
./hple-noCorr -data task2 -task reduce_label_noise/20 -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.25 -alpha 0.0001
./hple-noCorr -data task2 -task reduce_label_noise/30 -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.25 -alpha 0.0001
./hple-noCorr -data task2 -task reduce_label_noise/40 -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.25 -alpha 0.0001
./hple-noCorr -data task2 -task reduce_label_noise/50 -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.25 -alpha 0.0001
./hple-noCorr -data task2 -task reduce_label_noise/60 -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.25 -alpha 0.0001
./hple-noCorr -data task2 -task reduce_label_noise/70 -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.25 -alpha 0.0001
./hple-noCorr -data task2 -task reduce_label_noise/80 -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.25 -alpha 0.0001
./hple-noCor3 -data task2 -task reduce_label_noise/90 -mode bcd -size 50 -negatives 10 -iters 50 -threads 20 -lr 0.25 -alpha 0.0001


### testing
# echo 'NOT using candidates'
# echo '10% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test.py task2 reduce_label_noise/10 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '20% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test.py task2 reduce_label_noise/20 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '30% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test.py task2 reduce_label_noise/30 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '40% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test.py task2 reduce_label_noise/40 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '50% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test.py task2 reduce_label_noise/50 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '60% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test.py task2 reduce_label_noise/60 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '70% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test.py task2 reduce_label_noise/70 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '80% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test.py task2 reduce_label_noise/80 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '90% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test.py task2 reduce_label_noise/90 predict hple_corrH hete_feature maximum cosine 0.0



# echo 'Use candidates'
# echo '10% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test_cand.py task2 reduce_label_noise/10 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '20% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test_cand.py task2 reduce_label_noise/20 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '30% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test_cand.py task2 reduce_label_noise/30 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '40% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test_cand.py task2 reduce_label_noise/40 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '50% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test_cand.py task2 reduce_label_noise/50 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '60% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test_cand.py task2 reduce_label_noise/60 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '70% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test_cand.py task2 reduce_label_noise/70 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '80% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test_cand.py task2 reduce_label_noise/80 predict hple_corrH hete_feature maximum cosine 0.0
# echo ' '
# echo '90% '
# python /srv/data/wenqihe/Experiments/Evaluation/emb_test_cand.py task2 reduce_label_noise/90 predict hple_corrH hete_feature maximum cosine 0.0