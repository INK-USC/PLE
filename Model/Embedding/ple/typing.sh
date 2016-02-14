### dir: /srv/data/wenqihe/Experiments/Model/Embedding/hple/
echo 'data = ' $1 # FIGER, ontonotes, BBN
echo 'inference = ' $2 # maximum, topdown
echo 'sim metric = ' $3 # cosine, dot
echo 'threshold = ' $4
echo ''

### hple and hple-corr
echo 'Typing on Unseen Mentions'
echo ' '
echo 'hple_noCorr'
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_test.py $1 improve_typing/no predict hple_noCorr hete_feature $2 $3 $4
echo 'hple_corrH'
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_test.py $1 improve_typing/no predict hple_corrH hete_feature $2 $3 $4
echo 'hple_corrKB'
pypy /srv/data/wenqihe/Experiments/Evaluation/emb_test.py $1 improve_typing/no predict hple_corrKB hete_feature $2 $3 $4

# echo ' '
# echo 'Use candidate'
# echo ' '
# echo 'hple_noCorr'
# pypy /srv/data/wenqihe/Experiments/Evaluation/emb_test_cand.py $1 improve_typing/no predict hple_noCorr hete_feature $2 $3 $4
# echo 'hple_corrH'
# pypy /srv/data/wenqihe/Experiments/Evaluation/emb_test_cand.py $1 improve_typing/no predict hple_corrH hete_feature $2 $3 $4
# echo 'hple_corrKB'
# pypy /srv/data/wenqihe/Experiments/Evaluation/emb_test_cand.py $1 improve_typing/no predict hple_corrKB hete_feature $2 $3 $4

# echo ' '
# echo 'WSABIE'
# pypy /srv/data/wenqihe/Experiments/Evaluation/emb_test.py $1 improve_typing/no predict warp bipartite $2 $3 $4
