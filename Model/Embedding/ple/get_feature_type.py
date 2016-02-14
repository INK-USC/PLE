import sys
from collections import defaultdict
# import numpy as np
# from scipy.sparse import coo_matrix

def load_as_matrix(filename, col_number):
    """
    Load data as a sparse matrix.
    e.g.[[0,1,2],[1,2]]
    """
    with open(filename) as f:
        line = f.readline()
        seg = line.strip('\r\n').split('\t')
        k = 0
        length = 1
        index = int(seg[0])
        indexes = [index]
        row = [k]
        col = [int(seg[1])]
        for line in f:
            length += 1
            seg = line.strip('\r\n').split('\t')
            if index == int(seg[0]):  # Still in the same mention
                row.append(k)
                col.append(int(seg[1]))
            else:
                # Meet a new mention
                k += 1
                row.append(k)
                col.append(int(seg[1]))
                index = int(seg[0])
                indexes.append(index)
        data = np.array([1]*length)
        row = np.array(row)
        col = np.array(col)
        matrix = coo_matrix((data, (row, col)), shape=(len(indexes), col_number)).tocsr()
        return indexes, matrix



### input
feature = '/srv/data/wenqihe/boostrap/reduce_label_noise/feature.txt'
ttype = '/srv/data/wenqihe/boostrap/reduce_label_noise/type.txt'
mention_feature = '/srv/data/wenqihe/boostrap/reduce_label_noise/' + sys.argv[1]
mention_type = '/srv/data/wenqihe/boostrap/reduce_label_noise/' + sys.argv[2]

### output
feature_type = '/srv/data/wenqihe/boostrap/reduce_label_noise/' + sys.argv[3]

feature_number = 0
with open(feature) as f:
    for line in f:
        if line.strip():
            feature_number += 1

type_number = 0
with open(ttype) as f:
    for line in f:
        if line.strip():
            type_number += 1

### use numpy
# index1, MF = load_as_matrix(mention_feature, feature_number)
# index2, MT = load_as_matrix(mention_type, type_number)
# FT = MF.T.dot(MT)

# with open(feature_type, 'w') as g:
#     row, col = FT.nonzero()
#     length = row.shape[0]
#     for i in xrange(length):
#         r = row[i]
#         c = col[i]
#         g.write(str(r)+'\t'+str(c)+'\t'+str(FT[r, c])+'\n')


with open(mention_feature) as f1, open(mention_type) as f2,\
        open(feature_type,'w') as g:
        fm = defaultdict(set)
        tm = defaultdict(set)
        for line in f1:
            seg = line.strip('\r\n').split('\t')
            i = int(seg[0])
            j = int(seg[1])
            fm[j].add(i)
        for line in f2:
            seg = line.strip('\r\n').split('\t')
            i = int(seg[0])
            j = int(seg[1])
            tm[j].add(i)
        for i in xrange(feature_number):
            for j in xrange(type_number):
                temp = len(fm[i]&tm[j])
                if temp > 0:
                    g.write(str(i)+'\t'+str(j)+'\t'+str(temp)+'\n')

                    