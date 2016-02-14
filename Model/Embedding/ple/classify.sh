### dir: /srv/data/wenqihe/Experiments/Model/Embedding/hple/

echo 'classifier = ' $1 # perceptron, svm
echo 'data = ' $2 # FIGER, BBN, Ontonotes
echo 'lr for perceptron / C for SVM-liblinear = ' $3 # 0.003
echo 'max_iter = ' $4 # perceptron --- 1 for FIGER, 50 for ontonotes; SVM-pegasos---5000

echo ''
echo 'hple-noCorr'
pypy /srv/data/wenqihe/Experiments/Classifier/Classifier.py $1 $2/improve_typing/no hple_noCorr hete_feature $3 $4
echo 'hple-corrH'
pypy /srv/data/wenqihe/Experiments/Classifier/Classifier.py $1 $2/improve_typing/no hple_corrH hete_feature $3 $4
echo 'hple-corrKB'
pypy /srv/data/wenqihe/Experiments/Classifier/Classifier.py $1 $2/improve_typing/no hple_corrKB hete_feature $3 $4
