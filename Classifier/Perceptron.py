__author__ = 'wenqihe'
import sys



class MultilabelPerceptron:
    threshold = 0.3

    def __init__(self, feature_size, label_size, weights=None, learning_rate=0.003, max_iter=1):
        if weights is None:
            self._weights = [[0 for col in range(feature_size)] for row in range(label_size)]
        else:
            self._weights = weights
        self._feature_size = feature_size
        self._label_size = label_size
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        print 'max_iter = ', max_iter

    def fit(self, train_x, train_y):
        for time in xrange(self._max_iter):
            sys.stdout.write('{0} iteration done.\r'.format(time))
            sys.stdout.flush()
            for i in xrange(len(train_x)):
                x = train_x[i]
                y = train_y[i]
                predictions = self.predict(x)
                for l in predictions:
                    if l not in y:
                        for feature in x:
                            self._weights[l][feature] -= self._learning_rate
                for l in y:
                    if l not in predictions:
                        for feature in x:
                            self._weights[l][feature] += self._learning_rate
        print('Finish training.')

    def predict(self, x):
        labels = set()
        maxid = 0
        maxscore = -1
        for i in xrange(0, self._label_size):
            result = 0
            for feature in x:
                result += self._weights[i][feature]
            if result > MultilabelPerceptron.threshold:
                labels.add(i)
            if result > maxscore:
                maxid = i
                maxscore = result
        if len(labels) == 0:
            labels.add(maxid)
        return labels







    

