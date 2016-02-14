### dir: /srv/data/wenqihe/Experiments/Model/Embedding/hple/

echo 'data = ' $1 # FIGER, BBN, Ontonotes
# echo 'inference = ' $2 # maximum, topdown
# echo 'sim metric = ' $3 # cosine, dot
# echo 'threshold = ' $4

pypy /srv/data/wenqihe/Experiments/Evaluation/emb_prediction.py $1 improve_typing/no clean hple_noCorr hete_feature topdown dot -1
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_prediction.py $1 improve_typing/no clean hple_corrKB hete_feature topdown dot -1
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_prediction.py $1 improve_typing/no clean hple_corrH hete_feature topdown dot -1

# pypy /srv/data/wenqihe/Experiments/Evaluation/emb_prediction_cand.py $1 improve_typing/no clean hple_noCorr hete_feature maximum dot -100.0
# pypy /srv/data/wenqihe/Experiments/Evaluation/emb_prediction_cand.py $1 improve_typing/no clean hple_corrKB hete_feature maximum dot -100.0
# pypy /srv/data/wenqihe/Experiments/Evaluation/emb_prediction_cand.py $1 improve_typing/no clean hple_corrH hete_feature maximum dot -100.0