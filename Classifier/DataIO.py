__author__ = 'wenqihe'


def load_as_list(filename):
    """
    Load data as a list of list.
    e.g.[[0,1,2],[1,2]]
    """
    with open(filename) as f:
        data = []
        indexes = []
        line = f.readline()
        seg = line.strip('\r\n').split('\t')
        index = int(seg[0])
        features = [int(seg[1])]
        for line in f:
            seg = line.strip('\r\n').split('\t')
            if index == int(seg[0]):  # Still in the same mention
                features.append(int(seg[1]))
            else:
                # Append to train_x
                data.append(sorted(features))
                indexes.append(index)
                features = [int(seg[1])]
                index = int(seg[0])
        if len(features) > 0:
            data.append(sorted(features))
            indexes.append(index)
        return indexes, data


def save_from_list(filename, indexes, data):
    """
    Save data(a list of list) to a file.
    :param filename:
    :param data:
    :return:
    """
    with open(filename, 'w') as f:
        for i in xrange(len(indexes)):
            index = indexes[i]
            labels = data[i]
            for l in labels:
                f.write(str(index) + '\t' +str(l) + '\t1\n')

def load_as_dict(filename):
    with open(filename) as f:
        data = []
        indexes = []
        line = f.readline()
        seg = line.strip('\r\n').split('\t')
        index = int(seg[0])
        features = {(int(seg[1])+1): 1}
        for line in f:
            seg = line.strip('\r\n').split('\t')
            if index == int(seg[0]):  # Still in the same mention
                features[(int(seg[1])+1)] = 1
            else:
                # Append to train_x
                data.append(features)
                indexes.append(index)
                features = {(int(seg[1])+1): 1}
                index = int(seg[0])
        if len(features) > 0:
            data.append(features)
            indexes.append(index)
        return indexes, data


def file_len(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
