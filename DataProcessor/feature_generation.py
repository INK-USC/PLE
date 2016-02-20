__author__ = 'wenqihe'
from nlp_parse import parse
from ner_feature import pipeline, filter, pipeline_test
from statistic import supertype, distribution
from pruning_heuristics import prune


def get_number(filename):
    with open(filename) as f:
        count = 0
        for line in f:
            count += 1
        return count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print 'Usage:feature_generation.py -DATA'
        exit(1)
    indir = 'Data/%s' % sys.argv[1]
    outdir = 'Intermediate/%s' % sys.argv[1]
    # NLP parse
    raw_train_json = indir + '/train.json'
    raw_test_json = indir + '/test.json'
    train_json = outdir + '/train_new.json'
    test_json = outdir + '/test_new.json'

    # Generate features
    parse(raw_train_json, train_json)
    parse(raw_test_json, test_json)
    pipeline(JsonTrain, indir + '/brown', outdir)
    filter(outdir+'/feature.map', outdir+'/train_x.txt', outdir+'/feature.txt', outdir+'/train_x_new.txt')
    pipeline_test(JsonTest, indir + '/brown', outdir+'/feature.txt',outdir+'/type.txt', outdir)
    supertype(outdir)
    distribution(outdir)
    
    # Perform no pruning to generate training data
    feautre_number = get_number(outdir+'/feature.txt')
    type_number = outdir+'/type.txt'
    prune(outdir, outdir, 'no', feature_number, type_number)

    # Generate type type correlation
    share_entity(indir + '/type_entities.txt', outdir + '/type.txt', outdir + '/type_type_kb.txt')
