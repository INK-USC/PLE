# train-iter1
./hple-bs -data boostrap -task reduce_label_noise -boostrap 0 -mode bcd -size 50 -negatives 10 -iters 80 -threads 20 -lr 0.4 -alpha 0.0001
# predict-iter1
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_test.py boostrap reduce_label_noise predict hple_noCorr hete_feature maximum dot -100
# clean
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_boostrap_clean.py boostrap reduce_label_noise 1 hple_noCorr hete_feature maximum dot -100
# data update
pypy get_feature_type.py mention_feature.txt mention_type_1.txt feature_type_1.txt


# train-iter2
./hple-bs -data boostrap -task reduce_label_noise -boostrap 1 -mode bcd -size 50 -negatives 10 -iters 80 -threads 20 -lr 0.4 -alpha 0.0001
# predict-iter2
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_test.py boostrap reduce_label_noise predict hple_noCorr hete_feature maximum dot -100
# clean-iter2
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_boostrap_clean.py boostrap reduce_label_noise 2 hple_noCorr hete_feature maximum dot -100
# data update
pypy get_feature_type.py mention_feature.txt mention_type_2.txt feature_type_2.txt


# train-iter3
./hple-bs -data boostrap -task reduce_label_noise -boostrap 2 -mode bcd -size 50 -negatives 10 -iters 80 -threads 20 -lr 0.4 -alpha 0.0001
# predict-iter2
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_test.py boostrap reduce_label_noise predict hple_noCorr hete_feature maximum dot -100
# clean-iter2
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_boostrap_clean.py boostrap reduce_label_noise 3 hple_noCorr hete_feature maximum dot -100
# data update
pypy get_feature_type.py mention_feature.txt mention_type_3.txt feature_type_3.txt


# train-iter4
./hple-bs -data boostrap -task reduce_label_noise -boostrap 3 -mode bcd -size 50 -negatives 10 -iters 80 -threads 20 -lr 0.4 -alpha 0.0001
# predict-iter2
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_test.py boostrap reduce_label_noise predict hple_noCorr hete_feature maximum dot -100
# clean-iter2
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_boostrap_clean.py boostrap reduce_label_noise 4 hple_noCorr hete_feature maximum dot -100
# data update
pypy get_feature_type.py mention_feature.txt mention_type_4.txt feature_type_4.txt


# train-iter5
./hple-bs -data boostrap -task reduce_label_noise -boostrap 4 -mode bcd -size 50 -negatives 10 -iters 80 -threads 20 -lr 0.4 -alpha 0.0001
# predict-iter5
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_test.py boostrap reduce_label_noise predict hple_noCorr hete_feature maximum dot -100


