## PLE
Label Noise Reduction in Entity Typing by Heterogeneous Partial-Label Embedding

## Publication

* Xiang Ren\*, Wenqi He, Meng Qu, Clare R. Voss, Heng Ji, Jiawei Han, "**[Label Noise Reduction in Entity Typing by Heterogeneous Partial-Label Embedding](http://arxiv.org/abs/1602.05307)**”

## Requirements

We will take Ubuntu for example.

* python 2.7
```
$ sudo apt-get install python
```

* [stanford coreNLP 3.5.2](http://stanfordnlp.github.io/CoreNLP/) and its [python wrapper](https://github.com/dasmith/stanford-corenlp-python). Please put the library in folder DataProcessor/.

* [eigen 3.2.5](eigen.tuxfamily.org/). Please put the library in folder Model/ple/.

## Dataset
Three datasets used in the paper could be downloaded here:
   * FIGER
   * OntoNotes
   * BBN

Please put the data files in corresponding subdirectories in Data/.

## Default Run
Run PLE for the task of Reduce Label Noise on the BBN dataset

```
$ ./run.sh  
```

## Parameters - run.sh
Dataset to run on.
```
Data="BBN"
```

