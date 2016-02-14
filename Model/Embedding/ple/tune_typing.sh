### dir: /srv/data/wenqihe/Experiments/Model/Embedding/hple/
echo 'data = ' $1 # FIGER, ontonotes, BBN
echo 'method = ' $2 # hple_noCorr, hple_corrH, hple_corrKB

### hple and hple-corr
echo ' '
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_tune_threshold.py $1 improve_typing/no $2 hete_feature maximum dot
echo ' '
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_tune_threshold.py $1 improve_typing/no $2 hete_feature maximum cosine
echo ' '
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_tune_threshold.py $1 improve_typing/no $2 hete_feature topdown dot
echo ' '
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_tune_threshold.py $1 improve_typing/no $2 hete_feature topdown cosine