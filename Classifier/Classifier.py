__author__ = 'wenqihe'
import sys
from DataIO import *
from Perceptron import MultilabelPerceptron
from HierarchySVM3 import HierarchySVM3
from HierarchySVM import HierarchySVM
from PLSVM import PLSVM
from CLPL import CLPL
from TypeHierarchy import TypeHierarchy
from Evaluation.evaluation import evaluate, load_labels

def classify(model_name, feature_size, label_size, train_x, train_y, learning_rate, max_iter, type_hierarchy):
    if model_name == 'perceptron':
        model = MultilabelPerceptron(feature_size=_feature_size,
                                     label_size=_label_size,
                                     learning_rate=_learning_rate,
                                     max_iter=_max_iter)
    if model_name == 'svm':
        model = HierarchySVM3(feature_size=feature_size, type_hierarchy=type_hierarchy._subtype_mapping, current_types=type_hierarchy._root, level=0, threshold=0.0, C=learning_rate)
    if model_name == 'plsvm':        
        model = PLSVM(feature_size=feature_size, label_size=label_size, type_hierarchy=type_hierarchy, lambda_reg=0.1, max_iter=max_iter, threshold=0, batch_size=1000)
    if model_name == 'clpl':        
        model = CLPL(feature_size=feature_size, label_size=label_size, type_hierarchy=type_hierarchy, lambda_reg=0.1, max_iter=max_iter, threshold=10, batch_size=1000)
    if model_name == 'svm-pegasos':
        model = HierarchySVM(feature_size=feature_size, type_hierarchy=type_hierarchy._subtype_mapping, current_types=type_hierarchy._root, level=0, lambda_reg=learning_rate, max_iter=max_iter, threshold=-100)
    if model:
        model.fit(train_x, train_y)

    return model


def predict(model, test_x, type_hierarchy):
    MultilabelPerceptron.threshold = 0.35
    test_y = []
    type_distrubtion = {}
    for i in xrange(len(test_x)):
        x = test_x[i]
        labels = model.predict(x)
        parents = set()
        for l in labels:
            p = type_hierarchy.get_type_path(l)
            if len(p) > 1:
                parents.update(p)
        labels.update(parents)
        test_y.append(labels)
        for l in labels:
            if l in type_distrubtion:
                type_distrubtion[l]+=1
            else:
                type_distrubtion[l] = 1
    # print type_distrubtion
    return test_y


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print 'Usage: Classifier.py -CLASSIFIER -INDIR(FIGER/improve_typing/no) -METHOD(pte) -EMB_METHOD(bipartite) -LEARNING_RATE(0.003) -MAX_ITER(1)'
        exit(-1)
    model_name = sys.argv[1]
    indir = '/srv/data/wenqihe/' + sys.argv[2]
    train_x_file = indir + '/mention_feature.txt'
    train_y_file = indir + '/mention_type.txt'
    test_x_file = indir + '/mention_feature_test.txt'
    test_y_file = indir + '/Results/prediction_null_null_' + model_name + '.txt'
    hierarchy_file = indir + '/figer_supertype.txt'
    feature_file = indir + '/feature.txt'
    type_file = indir + '/type.txt'
    _method = sys.argv[3]
    _emb_method = sys.argv[4]
    _learning_rate = float(sys.argv[5])
    _max_iter = int(sys.argv[6])

    _feature_size = file_len(feature_file)
    _label_size = file_len(type_file)
    print 'Feature: %d, type: %d' %(_feature_size, _label_size)

    if _method != 'null' and _emb_method != 'null':
        train_y_file = indir + '/Results/mention_type_' + _method + '_' + _emb_method + '.txt'
        test_y_file = indir + '/Results/prediction_' + _method + '_' + _emb_method + '_' + model_name  + '.txt'

    train_x = load_as_list(train_x_file)
    train_y = load_as_list(train_y_file)
    ground_truth = load_labels(indir + '/mention_type_test.txt')

    assert len(train_x[1]) == len(train_y[1])
    print 'Total number of training examples: %d' % len(train_x[1])
    print 'Start training'
    type_hierarchy = TypeHierarchy(hierarchy_file, _label_size)
    model = classify(model_name, _feature_size, _label_size, train_x[1], train_y[1], _learning_rate, _max_iter, type_hierarchy)
    indexes, test_x = load_as_list(test_x_file)
    test_y = predict(model, test_x, type_hierarchy)
    save_from_list(test_y_file, indexes, test_y)


    ### Evluate embedding predictions
    predictions = load_labels(test_y_file)
    print 'Predicted labels ('+model_name+'):'
    print 'Test mentions:%d,%d'%(len(predictions),len(ground_truth))
    print '%f\t%f\t%f\t%f\t%f\t%f\t%f\t' % evaluate(predictions, ground_truth)



