
echo 'method = ' $1
echo 'emb_mode = ' $2 # bipartite, hete_feature
echo 'predict_mode = ' $3 # maximum, topdown
echo 'sim = ' $4 # dot, cosine
echo 'threshold = ' $5 

pypy /srv/data/wenqihe/Experiments/Evaluation/evaluation_hierarchy.py reduce_label_noise/no $1 $2 $3 $4 $5