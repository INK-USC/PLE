__author__ = 'wenqihe'
import sys
import json
from DataIO import *
from Perceptron import MultilabelPerceptron
from HierarchySVM import HierarchySVM
from PLSVM import PLSVM
from CLPL import CLPL
from TypeHierarchy import TypeHierarchy

def classify(model_name, feature_size, label_size, train_x, train_y, learning_rate, max_iter, type_hierarchy):
    if model_name == 'perceptron':
        model = MultilabelPerceptron(feature_size=_feature_size,
                                     label_size=_label_size,
                                     learning_rate=_learning_rate,
                                     max_iter=_max_iter)
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


def casestudy(filename, output, mention_mapping, label_mapping, clean_mentions):
    with open(filename) as f, open(output, 'w') as g:
        for line in f:
            sent = json.loads(line.strip('\r\n'))
            result = putback(sent, mention_mapping, label_mapping, clean_mentions)
            if result is not '':
                g.write(result+'\n')


def putback(sent_json, mention_mapping, label_mapping, clean_mentions):
    fileid = sent_json['fileid']
    senid = sent_json['senid']
    tokens = sent_json['tokens']
    pivot = 0
    result = []
    mentions = sent_json['mentions']
    sorted_m = sorted(mentions, cmp=compare)
    for m in sorted_m:
        start = m['start']
        end = m['end']
        if end - start == 1:
            mention_name = '[%s]' % (tokens[start])
        else:
            mention_name = '[%s]' % (' '.join(tokens[start:end]))
        if pivot <= start:
            result.extend(tokens[pivot:start])
            result.append(mention_name)
            # find predicted labels if any
            m_name = '%s_%d_%d_%d'%(fileid, senid, start, end)
            if m_name in mention_mapping:
                m_id = mention_mapping[m_name]
                if m_id in clean_mentions:
                    clean_labels = [label_mapping[l] for l in clean_mentions[m_id]]
                    result.append(':'+'['+','.join(clean_labels)+']')
        pivot = end
    if pivot < len(tokens):
        result.extend(tokens[pivot:])
    result = ' '.join([x for x in result if x is not None])
    return fileid+':'+str(senid)+'\t'+result + '\n'

def compare(item1, item2):
    if item1['start'] != item2['start']:
        return item1['start'] - item2['start']
    else:
        return item2['end'] - item1['end']


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print 'Usage: Classifier.py -CLASSIFIER -DATA(BBN) -METHOD(pte) -EMB_METHOD(bipartite) -LEARNING_RATE(0.003) -MAX_ITER(20)'
        exit(-1)
    model_name = sys.argv[1]
    indir = 'Intermediate/' + sys.argv[2]
    train_x_file = indir + '/mention_feature.txt'
    train_y_file = indir + '/mention_type.txt'
    test_x_file = indir + '/mention_feature_test.txt'
    test_y_file = 'Results/'+ sys.argv[2]+'/prediction_null_null_' + model_name + '.txt'
    hierarchy_file = indir + '/supertype.txt'
    feature_file = indir + '/feature.txt'
    type_file = indir + '/type.txt'
    mention_file = indir + '/mention.txt'
    json_file = 'Data/' + sys.argv[2] + '/test.json'
    output = 'Results/'+ sys.argv[2]+'/predictionInText_null_null_' + model_name + '.txt'
    _method = sys.argv[3]
    _emb_method = sys.argv[4]
    _learning_rate = float(sys.argv[5])
    _max_iter = int(sys.argv[6])

    _feature_size = file_len(feature_file)
    _label_size = file_len(type_file)
    print 'Feature: %d, type: %d' %(_feature_size, _label_size)

    if _method != 'null' and _emb_method != 'null':
        train_y_file = 'Results/'+ sys.argv[2]+'/mention_type_' + _method + '_' + _emb_method + '.txt'
        test_y_file = 'Results/'+ sys.argv[2]+'/prediction_' + _method + '_' + _emb_method + '_' + model_name  + '.txt'
        output = 'Results/'+ sys.argv[2]+'/predictionInText_' + _method + '_' + _emb_method + '_' + model_name  + '.txt'

    train_x = load_as_list(train_x_file)
    train_y = load_as_list(train_y_file)

    ### Train
    assert len(train_x[1]) == len(train_y[1])
    print 'Total number of training examples: %d' % len(train_x[1])
    print 'Start training'
    type_hierarchy = TypeHierarchy(hierarchy_file, _label_size)
    model = classify(model_name, _feature_size, _label_size, train_x[1], train_y[1], _learning_rate, _max_iter, type_hierarchy)

    ### Test
    indexes, test_x = load_as_list(test_x_file)
    test_y = predict(model, test_x, type_hierarchy)
    save_from_list(test_y_file, indexes, test_y)

    ### Write inText Results
    mention_mapping = load_map(mention_file, 'mention')
    label_mapping = load_map(type_file, 'label')
    clean_mentions = load_mention_type(test_y_file)
    casestudy(json_file, output, mention_mapping, label_mapping, clean_mentions)





