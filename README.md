## PLE
Label Noise Reduction in Entity Typing by Heterogeneous Partial-Label Embedding

## Publication

* Xiang Ren\*, Wenqi He, Meng Qu, Clare R. Voss, Heng Ji, Jiawei Han, "**[Label Noise Reduction in Entity Typing by Heterogeneous Partial-Label Embedding](http://web.engr.illinois.edu/~xren7/kdd16-LNR.pdf)**”, SIGKDD 2016.

## Requirements

We will take Ubuntu for example.

* python 2.7
```
$ sudo apt-get install python
```

* [stanford coreNLP 3.7.0](http://stanfordnlp.github.io/CoreNLP/) and its [python wrapper](https://github.com/stanfordnlp/stanza). Please put the library in folder DataProcessor/.

* [eigen 3.2.5](eigen.tuxfamily.org/). Please put the library in folder Model/ple/.

* cd /Model/ple and "make"

## Dataset
Three datasets used in the paper could be downloaded here:
   * [Wiki](https://drive.google.com/file/d/0B2ke42d0kYFfVC1fazdKYnVhYWs/view?usp=sharing)
   * [OntoNotes](https://drive.google.com/file/d/0B2ke42d0kYFfN1ZSVExLNlYwX1E/view?usp=sharing)
   * [BBN](https://drive.google.com/file/d/0B2ke42d0kYFfTEs0RGpuanRLQlE/view?usp=sharing)

Please put the data files in corresponding subdirectories in Data/.

## Default Run
Run PLE for the task of Reduce Label Noise on the BBN dataset

```
$ java -mx4g -cp "DataProcessor/stanford-corenlp-full-2016-10-31/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
$ ./run.sh  
```

## Parameters - run.sh
Dataset to run on.
```
Data="BBN"
```

