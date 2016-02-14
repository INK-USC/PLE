### dir: /srv/data/wenqihe/Experiments/Model/Embedding/hple/
echo 'data = ' $1 # FIGER, onotonotes, BBN
echo 'method = ' $2 # hple, hple_corr, hple_corrH, hple_corrKB, hple_noCorr

### hple and hple-corr
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_tune_threshold.py $1 reduce_label_noise/no $2 hete_feature maximum dot
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_tune_threshold.py $1 reduce_label_noise/no $2 hete_feature maximum cosine
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_tune_threshold.py $1 reduce_label_noise/no $2 hete_feature topdown dot
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_tune_threshold.py $1 reduce_label_noise/no $2 hete_feature topdown cosine
echo ' '

### tune_threshold_cand
echo 'Use candidate set:'
echo ' '
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_tune_threshold_cand.py $1 reduce_label_noise/no $2 hete_feature maximum dot
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_tune_threshold_cand.py $1 reduce_label_noise/no $2 hete_feature maximum cosine
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_tune_threshold_cand.py $1 reduce_label_noise/no $2 hete_feature topdown dot
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_tune_threshold_cand.py $1 reduce_label_noise/no $2 hete_feature topdown cosine